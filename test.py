import argparse
import torch
from torch import nn, Tensor
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from torch.utils.data import DataLoader
from torch import Generator
from torch.utils.data import random_split
from torcheval.metrics.text import BLEUScore as TE_BLEU
# Your imports
from train import Transformer, TransformerConfig
from attention import attnconfig, AttnVariant, PositionalVariant
from encoder import EncoderConfig
from decoder import DecoderConfig
from utils import EnFinDataModule, EnFinnishDataset,CleanedEnFinnishDataset


class Strategies(Enum):
    Greedy = "greedy"
    TopK = "topk"
    Beam = "beam"
    
def _pad_bool_matrix(pad_idx: int, L: int) -> torch.Tensor:
    m = torch.zeros((L, L), dtype=torch.bool)
    if pad_idx < L:
        m[:, pad_idx:] = True
        m[pad_idx:, :] = True
    return m

class AutoRegressiveGenerator(nn.Module):
    def __init__(self, *, tokenizer, model, eos_token_id: Optional[int] = None, context_len: int = 512):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.eos_id = eos_token_id if eos_token_id is not None else getattr(tokenizer, "eos_token_id", None)
        self.context_len = int(context_len)

    def return_masks(self, pad_idx: int):
        pad_masks = torch.zeros(size=(self.context_len, self.context_len), device=next(self.model.parameters()).device)
        pad_masks[:, pad_idx:] = -1e9
        pad_masks[pad_idx:, :] = -1e9
        return pad_masks

    def step_select(self, logits_last: Tensor) -> Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def generate(
        self,
        en_tokens: Tensor,     # [B, S]
        en_mask: Tensor,       # broadcastable mask for encoder
        max_new_tokens: int = 64,
        bos_token_id: Optional[int] = None,
        temperature: float = 1.0,
        adding_padding:int=0,
    ) -> Tensor:
        """Greedy/TopK path uses this default implementation."""
        device = en_tokens.device
        self.model.eval()

        B = en_tokens.size(0)
        bos_id = bos_token_id if bos_token_id is not None else getattr(self.tokenizer, "bos_token_id", None)
        pad_id = self.tokenizer.pad_token_id
        assert bos_id is not None, "Need a BOS token id"

        fin_tokens = torch.full((B, self.context_len), pad_id, dtype=torch.long, device=device)
        fin_tokens[:, 0] = bos_id

        steps = min(max_new_tokens, self.context_len - 1)
        for i in range(1, steps + 1):
            if(adding_padding):
                fin_mask = self.return_masks(i).unsqueeze(0)
            else:
                fin_mask = _pad_bool_matrix(i,self.context_len).expand(en_tokens.shape[0],-1,-1)
            
            # print(f"Shape of en_tokens:{en_tokens.shape}")
            # print(f"Shape of en_mask:{en_mask.shape}")

            logits = self.model(en_tokens, fin_tokens, en_mask, fin_mask)  # [B, T, V]
            logits_last = logits[:, i, :]                                   # [B, V]

            if temperature and temperature != 1.0:
                logits_last = logits_last / temperature

            next_ids = self.step_select(logits_last)                         # [B]
            next_ids = next_ids.to(device=device, dtype=torch.long)
            fin_tokens[:, i] = next_ids

            if self.eos_id is not None and torch.all(next_ids == self.eos_id):
                break

        return fin_tokens

    def decode_text(self, token_batch: Tensor) -> list[str]:
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in token_batch.tolist()]

    def _causal_mask(self, t: int, device: torch.device) -> Optional[Tensor]:
        m = torch.full((t, t), float("-inf"), device=device)
        m = torch.triu(m, diagonal=1)
        return m.unsqueeze(0)  # [1, T, T]


class GreedyDecoding(AutoRegressiveGenerator):
    def step_select(self, logits_last: Tensor) -> Tensor:
        return torch.argmax(logits_last, dim=-1)


class TopKSampling(AutoRegressiveGenerator):
    def __init__(self, *, tokenizer, model, k: int = 50, eos_token_id: Optional[int] = None, context_len: int = 512):
        super().__init__(tokenizer=tokenizer, model=model, eos_token_id=eos_token_id, context_len=context_len)
        assert k >= 1
        self.k = int(k)

    def step_select(self, logits_last: Tensor) -> Tensor:
        topk_vals, topk_idx = torch.topk(logits_last, k=min(self.k, logits_last.size(-1)), dim=-1)  # [B, K]
        probs = torch.softmax(topk_vals, dim=-1)                                                    # [B, K]
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)                               # [B]
        return topk_idx.gather(dim=1, index=sampled.unsqueeze(1)).squeeze(1)                        # [B]


class BeamSearchDecoding(AutoRegressiveGenerator):
    """
    Batch-1 beam search with length penalty.
    score = sum(log p) / ( (5+L)^alpha / (5+1)^alpha )
    """
    def __init__(
        self,
        *,
        tokenizer,
        model,
        beam_size: int = 4,
        alpha: float = 0.6,
        eos_token_id: Optional[int] = None,
        context_len: int = 512,
    ):
        super().__init__(tokenizer=tokenizer, model=model, eos_token_id=eos_token_id, context_len=context_len)
        assert beam_size >= 1
        self.beam_size = int(beam_size)
        self.alpha = float(alpha)

    @torch.no_grad()
    def generate(
        self,
        en_tokens: Tensor,   # [1, S] (asserted)
        en_mask: Tensor,
        max_new_tokens: int = 64,
        bos_token_id: Optional[int] = None,
        temperature: float = 1.0,  # not used; could be supported
        adding_padding:int=0,
    ) -> Tensor:
        device = en_tokens.device
        self.model.eval()
        assert en_tokens.size(0) == 1, "BeamSearchDecoding currently supports batch_size=1."

        bos_id = bos_token_id if bos_token_id is not None else getattr(self.tokenizer, "bos_token_id", None)
        pad_id = self.tokenizer.pad_token_id
        assert bos_id is not None, "Need a BOS token id"

        T = self.context_len
        steps = min(max_new_tokens, T - 1)
        # Beams: tokens [B*, T], scores [B*]
        beams = torch.full((self.beam_size, T), pad_id, dtype=torch.long, device=device)
        beams[:, 0] = bos_id
        beam_scores = torch.zeros(self.beam_size, device=device)
        # Start with a single live beam; others are -inf so they won't be chosen
        beam_scores[1:] = float("-inf")

        finalized = []  # list of (score, tokens)

        for i in range(1, steps + 1):
            live_count = beams.size(0)

            # Expand encoder inputs across beams
            en_rep = en_tokens.expand(live_count, -1)
            en_mask_rep = en_mask.expand(live_count, *([-1] * (en_mask.dim() - 1))) if en_mask is not None else None

            if(adding_padding):
                fin_mask = self.return_masks(i).unsqueeze(0)
            else:
                fin_mask = _pad_bool_matrix(i,self.context_len).expand(en_rep.shape[0],-1,-1)
            
            logits = self.model(en_rep, beams, en_mask_rep, fin_mask)  # [live, T, V]
            logits_last = logits[:, i, :]                               # [live, V]
            logprobs = torch.log_softmax(logits_last, dim=-1)           # [live, V]

            # Sum with previous scores
            cand_scores = beam_scores.unsqueeze(1) + logprobs           # [live, V]
            cand_scores = cand_scores.view(-1)                          # [live*V]

            # Select top-K overall
            topk_scores, topk_indices = torch.topk(cand_scores, k=min(self.beam_size, cand_scores.numel()))
            # Map flat indices back to (beam_id, token_id)
            prev_beam_ids = topk_indices // logprobs.size(-1)
            token_ids = topk_indices % logprobs.size(-1)

            # Prepare next-beam buffers
            next_beams = torch.full_like(beams[: self.beam_size], pad_id)
            next_beam_scores = torch.full_like(beam_scores[: self.beam_size], float("-inf"))

            next_slot = 0
            for j in range(topk_scores.size(0)):
                b_id = prev_beam_ids[j].item()
                tok = token_ids[j].item()
                score = topk_scores[j].item()

                cand = beams[b_id].clone()
                cand[i] = tok

                if self.eos_id is not None and tok == self.eos_id:
                    # Length-penalized final score
                    L = i  # number of generated tokens including EOS at pos i
                    lp = ((5 + L) ** self.alpha) / ((5 + 1) ** self.alpha)
                    norm_score = score / lp
                    finalized.append((norm_score, cand.clone()))
                    continue

                capacity = next_beams.size(0)          # << key change
                if next_slot < capacity:
                    next_beams[next_slot] = cand
                    next_beam_scores[next_slot] = score
                    next_slot += 1

            # If we didn't fill beams (e.g., many EOS), carry over best survivors
            if next_slot == 0:
                # No survivors; must stop
                break

            beams = next_beams[:next_slot]
            beam_scores = next_beam_scores[:next_slot]

            # Early stop if we already have enough finals and no more room
            if len(finalized) >= self.beam_size and next_slot == 0:
                break

        # If nothing finalized, finalize best current beams
        if not finalized:
            for k in range(beams.size(0)):
                L = (beams[k] != self.tokenizer.pad_token_id).sum().item()
                L = max(L, 1)
                lp = ((5 + L) ** self.alpha) / ((5 + 1) ** self.alpha)
                finalized.append((beam_scores[k].item() / lp, beams[k].clone()))

        # Pick best
        finalized.sort(key=lambda x: x[0], reverse=True)
        best_tokens = finalized[0][1].unsqueeze(0)  # [1, T]
        return best_tokens


def build_configs(tokenizer, context_len=512, model_dim=512, num_heads_dec=4, num_heads_enc=4, n_blocks=8,post_pre_norm=0,
                  posn_class:int=1,adding_padding:int=0,
                  attn_class:AttnVariant=AttnVariant.FAST_MULTIHEADED,
                  ):
    if posn_class !=2:
        decoder_cfg = DecoderConfig(
            num_heads=num_heads_dec,
            vocab_size=len(tokenizer),
            embedding_size=model_dim,
            max_seq_len=context_len,
            atn_cfg=attnconfig(
                model_dim, model_dim, model_dim, model_dim, num_heads_dec, False, context_len,
                posn_class=PositionalVariant(1),adding_padding=1
            ),
            mlp_depth=2,
            attn_class=AttnVariant.FAST_MULTIHEADED,
            post_pre_norm=post_pre_norm,
        )
        encoder_cfg = EncoderConfig(
            num_heads=num_heads_enc,
            vocab_size=len(tokenizer),
            embedding_size=model_dim,
            max_seq_len=context_len,    
            atn_cfg=attnconfig(
                model_dim, model_dim, model_dim, model_dim, num_heads_dec, False, context_len,
                posn_class=PositionalVariant(1),adding_padding=1
            ),
            attn_class=AttnVariant.FAST_SELFMHA,
            # pos_weight=0.2,
            mlp_depth=2,
            post_pre_norm=post_pre_norm,
        )
    else:
        decoder_cfg = DecoderConfig(
            num_heads=num_heads_enc,
            # vocab_size=len(dm.train_ds.dataset.tokenizer),
            vocab_size=tokenizer.vocab_size,
            
            embedding_size=model_dim,
            max_seq_len=context_len,
            atn_cfg=attnconfig(query_dim=model_dim, key_dim=model_dim,value_dim= model_dim,model_dim= model_dim,
                            n_heads=num_heads_enc,causal_mask= False,context_len= context_len,
                            posn_class=PositionalVariant.NONE,posn_weight=0,adding_padding=adding_padding),
            attn_class=AttnVariant.RELATIVE_PE,
            # posn_class=PositionalVariant(1),
            mlp_depth=2,
            post_pre_norm=post_pre_norm
        )
        encoder_cfg = EncoderConfig(
            num_heads=4,
            # vocab_size=len(dm.train_ds.dataset.tokenizer),
            vocab_size=tokenizer.vocab_size,
            
            embedding_size=model_dim, ## this the dimensions of the output of the encoder
            max_seq_len=context_len,
            atn_cfg=attnconfig(query_dim=model_dim,key_dim= model_dim,value_dim= model_dim,model_dim= model_dim,
                            n_heads=num_heads_enc,causal_mask= False,context_len= context_len,
                            posn_class=PositionalVariant.NONE,posn_weight=0,adding_padding=adding_padding),
            # pos_weight=0.2,
            mlp_depth=2,
            attn_class=AttnVariant.RELATIVE_PE,  # Assuming 4 corresponds to a FastMHA variant
            # posn_class=PositionalVariant(1),
            post_pre_norm=post_pre_norm
        )
    transformer_cfg = TransformerConfig(
        n_blocks=n_blocks,
        encoder_cfg=encoder_cfg,
        decoder_cfg=decoder_cfg,
        tokenizer=tokenizer,
    )
    return transformer_cfg


def choose_generator(strategy: Strategies, tokenizer, model, context_len: int, topk: int, beam: int, alpha: float):
    if strategy == Strategies.Greedy:
        return GreedyDecoding(tokenizer=tokenizer, model=model, context_len=context_len)
    if strategy == Strategies.TopK:
        return TopKSampling(tokenizer=tokenizer, model=model, k=topk, context_len=context_len)
    if strategy == Strategies.Beam:
        return BeamSearchDecoding(tokenizer=tokenizer, model=model, beam_size=beam, alpha=alpha, context_len=context_len)
    raise ValueError(f"Unknown strategy: {strategy}")

from tqdm import tqdm
@torch.no_grad()
def run_decode(
    generator: AutoRegressiveGenerator,
    dataloader: DataLoader,
    tokenizer,
    device,
    max_new_tokens: int = 128,
    n_print_batches: int = 5,  # just for preview; BLEU is over the full dataset
    adding_padding:int=0
    ):
    preds = []
    refs  = []
    printed = 0

    with tqdm(total=len(dataloader),desc="Running Inference: ") as pbar:
        for en_tokens, en_mask, fin_tokens, fin_mask in dataloader:
            fin_tokens = fin_tokens.to(device)
            fin_mask = fin_mask.to(device) if fin_mask is not None else None
            # print(fin_mask,fin_mask.dtype)
            # Generate for the whole batch
            gen_ids = generator.generate(en_tokens=fin_tokens, en_mask=fin_mask, max_new_tokens=max_new_tokens,adding_padding=adding_padding)  # [B, T_out]
            B = gen_ids.size(0)
            for i in range(B):
                gen_text = tokenizer.decode(gen_ids[i].tolist(), skip_special_tokens=True)
                ref_text = tokenizer.decode(en_tokens[i].tolist(), skip_special_tokens=True)
                preds.append(gen_text)
                refs.append(ref_text)

            # Print a few samples for inspection
            if printed < n_print_batches:
                print(f"Machine translated output: {preds[-B]}")
                print(f"Original output:           {refs[-B]}")
                print("-" * 80)
                printed += 1
            pbar.update(1)
    metric = TE_BLEU(n_gram=4)
    metric.update(preds, [[r] for r in refs])
    out = metric.compute()
    bleu = float(out.item() if hasattr(out, "item") else out)
    print(bleu)
    print(f"[BLEU-4] Corpus BLEU over entire dataset: {bleu:.4f}")


    return {"bleu": bleu, "preds": preds, "refs": refs}
import torch

def pick_cuda_device(min_free_mem_gb: float = 0.0) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    best_idx, best_free_gb = None, -1.0
    for i in range(torch.cuda.device_count()):
        # Query free/total memory on each device
        with torch.cuda.device(i):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024 ** 3)

        if free_gb > best_free_gb and free_gb >= min_free_mem_gb:
            best_idx, best_free_gb = i, free_gb

    if best_idx is None:
        return torch.device("cpu")

    torch.cuda.set_device(best_idx)  # set default CUDA device for new tensors
    return torch.device(f"cuda:{best_idx}")

# Usage



def main():
    parser = argparse.ArgumentParser(description="En→Fi decoding with Greedy/TopK/Beam.")
    parser.add_argument("--archive_path", type=str, default="/kaggle/input/eng-fin-translation-dataset")
    parser.add_argument("--model_path", type=str, default="/kaggle/input/my_model/pytorch/2b/1/best-model-epoch16-val_loss5.45.ckpt")
    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--num_heads_dec", type=int, default=8)
    parser.add_argument("--num_heads_enc", type=int, default=4)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--post_pre_norm", type=int, default=1)
    parser.add_argument('--posn_class', type=int,default=1,help="1: RoPE, 2: RelativePE")
    parser.add_argument('--adding_padding',type=int,default=0,help="Argument to switch between adding or boolean masking padding. O for not adding. 1 for adding")
    parser.add_argument('--attn_class',type=int,default=3,help="Whether to pad mask for keys only or full pad mask.")
    parser.add_argument('--keys_only',type=int,default=1,help="Whether to pad mask for keys only or full pad mask.")

    
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_batches", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=128)

    parser.add_argument("--strategy", type=str, choices=[s.value for s in Strategies], default="greedy")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.6)

    args = parser.parse_args()

    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    device = pick_cuda_device(min_free_mem_gb=2.0)  # e.g., require at least 2 GB free
    print("Using device:", device)
    # Dataset
    FullDataset = CleanedEnFinnishDataset(args.archive_path, context_len=args.context_len,
                                          adding_padding=args.adding_padding,keys_only=args.keys_only)
    train_test = 0.999
    train_val=0.8
    ##same seed used at training.
    full_len = len(FullDataset)
    train_len = int(train_test * full_len)
    test_len = full_len - train_len
    val_len = int((1-train_val) * train_len)
    train_len = train_len - val_len
    generator = Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(
        FullDataset,
        [train_len, val_len, test_len],
        generator=generator
    )
    dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Configs + model
    transformer_cfg = build_configs(
        tokenizer=FullDataset.tokenizer,
        context_len=args.context_len,
        model_dim=args.model_dim,
        num_heads_dec=args.num_heads_dec,
        num_heads_enc=args.num_heads_enc,
        n_blocks=args.n_blocks,
        post_pre_norm=args.post_pre_norm,
        posn_class=args.posn_class,
        adding_padding=args.adding_padding,
        attn_class=args.attn_class
    )

    litmodel = Transformer.load_from_checkpoint(args.model_path, config=transformer_cfg)
    litmodel.eval().to(device)
    underlying_model = getattr(litmodel, "model", litmodel)

    # Generator
    strategy = Strategies(args.strategy)
    generator = choose_generator(
        strategy=strategy,
        tokenizer=FullDataset.tokenizer,
        model=underlying_model,
        context_len=args.context_len,
        topk=args.topk,
        beam=args.beam,
        alpha=args.alpha,
    ).to(device)

    # Decode
    run_decode(
        generator=generator,
        dataloader=dataloader,
        tokenizer=FullDataset.tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
        n_print_batches=args.n_batches,
    )


if __name__ == "__main__":
    main()
