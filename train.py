from encoder import EncoderConfig,TransformerEncoderBlock,TransformerEncoder
from decoder import DecoderConfig,TransformerDecoderBlock,TransformerDecoder
from dataclasses import dataclass,field
from utils import *
import torch
from torch import Tensor,nn
import lightning
from attention import attnconfig,AttnVariant,PositionalVariant
from typing import Optional
from lightning import Trainer
import wandb
from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint,RichModelSummary
from lightning.pytorch.loggers import WandbLogger
import argparse 
from torcheval.metrics.text import BLEUScore
@dataclass 
class TransformerConfig:
    encoder_cfg:EncoderConfig
    decoder_cfg:DecoderConfig
    tokenizer:GPT2Tokenizer
    n_blocks:int=4
    decoding_strategy:int = 1
    topk:int=6

class Transformer(lightning.LightningModule):
    def __init__(self, config:TransformerConfig,logval_everyk =50):
        super().__init__()
        self.encoder =  TransformerEncoder(config.encoder_cfg,config.n_blocks)
        self.decoder =  TransformerDecoder(config.decoder_cfg,config.n_blocks)
        self.tokenizer = config.tokenizer
        self.loss =  torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id,label_smoothing=0.1)
        self.bleu = BLEUScore(n_gram=4) ## make this configurable 
        self.logval_everyk = logval_everyk
        self.topk = config.topk
        self.decoding_strategy = config.decoding_strategy
        self.context_length  = config.encoder_cfg.max_seq_len
    def forward(self,eng_tokens,fin_tokens,en_mask:Optional[Tensor],fin_mask:Optional[Tensor]):
        encoder_outputs = self.encoder(eng_tokens,en_mask)        
        decoder_output = self.decoder(fin_tokens,encoder_outputs,en_mask,fin_mask)
        return decoder_output
    def training_step(self,batch,batch_idx):
        en_tokens,en_mask,fin_tokens,fin_mask = batch
        probs = self(en_tokens,fin_tokens,en_mask,fin_mask)
        probs = probs[...,:-1,:].reshape(-1,probs.shape[-1]) ## last dim is of size vocab_size, So now (batch x num_tokens), vocab_size tensor. (2-D) tensor
        loss = self.loss(probs,fin_tokens[:,1:].reshape(-1)) ## this will now be batch_size x sequence_length long; (1-D) tensor
    ## Start from token 2 onwards, as i want it to predict these tokens, and drop the last logit, as i dont care for the output given the entire sequence
        self.log("train_loss", loss, prog_bar=True,on_step=True,on_epoch=True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        en_tokens,en_mask,fin_tokens,fin_mask = batch
        probs = self(en_tokens,fin_tokens,en_mask,fin_mask)  ## to not give it the last token, which i want it to predict.
        probs = probs[...,:-1,:].reshape(-1,probs.shape[-1]) ## last dim is of size vocab_size, So now (batch x num_tokens), vocab_size tensor. (2-D) tensor
        val_loss = self.loss(probs,fin_tokens[:,1:].reshape(-1)) ## this will now be batch_size x sequence_length long; (1-D) tensor
        
        self.log("val_loss",val_loss,prog_bar=True,on_step=True,on_epoch=True)
        # if batch_idx % self.logval_everyk == 0:
        #     metrics = self.predict_step(batch)
        #     self.log_dict(metrics,on_step=True,on_epoch=False)        
    ## Start from token 2 onwards, as i want it to predict these tokens, and drop the last logit, as i dont care for the output given the entire sequence
        return val_loss
    def log_grad_norms(self):
        grad_norms = {}
        total_norm = 0.0
        # Use self.named_parameters() to get the model's parameters
        for name, p in self.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.norm(2)
                total_norm += param_norm.item() ** 2
                grad_norms[f"grad_norm/{name}"] = param_norm.item()

        total_norm = total_norm ** 0.5
        grad_norms["grad_norm/total"] = total_norm        
        self.log_dict(grad_norms, on_step=True, on_epoch=False)
        
    def on_after_backward(self):
        self.log_grad_norms()
    
    def return_masks(self, t: int, device):
        # causal additive mask for length t (decoder self-attn)
        # shape [1, t, t] with -1e9 above diagonal
        m = torch.full((t, t), 0.0, device=device)
        m = torch.triu(m, diagonal=1)  # 1 above diagonal
        m[m == 1] = -1e9
        return m.unsqueeze(0)  # [1, t, t]

    def greedy_choice(self,logits:Optional[Tensor]):
        return(torch.argmax(logits,dim=-1)) 

    def topk_choice(self,logits:Optional[Tensor]):
        topk_vals, topk_idx = torch.topk(logits, k=min(self.topk, logits.size(-1)), dim=-1)  # [B, K]
        probs = torch.softmax(topk_vals, dim=-1)  # [B, K]
        sampled_in_topk = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]
        next_ids = topk_idx.gather(dim=1, index=sampled_in_topk.unsqueeze(1)).squeeze(1)  # [B]
        return next_ids


    def choose_index(self,logits:Tensor):
        if self.decoding_strategy == 1:
            return self.greedy_choice(logits=logits)
        elif self.decoding_strategy == 2:
            return self.topk_choice(logits=logits)
    
    @torch.no_grad()
    def generate(self, english_tokens, english_mask, max_length: int = 512):
        device = english_tokens.device
        B = english_tokens.size(0)
        bos = getattr(self.tokenizer, "bos_token_id",
                    getattr(self.tokenizer, "cls_token_id", self.tokenizer.eos_token_id))
        pad = self.tokenizer.pad_token_id
        fin_tokens = torch.full((B, max_length), pad, dtype=torch.long, device=device)
        fin_tokens[:, 0] = bos
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        for t in range(1, max_length):
            fin_mask = self.return_masks(t, device)             # [1, t, t]
            logits = self(english_tokens, fin_tokens[:, :t], english_mask, fin_mask)  # (B, t, V)
            next_logits = logits[:, -1, :]                       # (B, V)
            if self.decoding_strategy == 1:
                next_ids = torch.argmax(next_logits, dim=-1)
            else:
                topk_vals, topk_idx = torch.topk(next_logits, k=min(self.topk, next_logits.size(-1)), dim=-1)
                probs = torch.softmax(topk_vals, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
                next_ids = topk_idx.gather(1, sampled.unsqueeze(1)).squeeze(1)
            fin_tokens[:, t] = next_ids
            if eos_id is not None and (next_ids == eos_id).all():
                break

        return fin_tokens

    
        
    def predict_step(self,batch):
        en_tokens,en_mask,fin_tokens,fin_mask = batch
        mtlfin_tokens =  self.generate(english_tokens=en_tokens,
                                         english_mask=en_mask,
                                         max_length=self.context_length)
        mtlfin_text = self.tokenizer.batch_decode(mtlfin_tokens)
        truefin_text = self.tokenizer.batch_decode(fin_tokens)
        metrics = {
            "valbleu_score": self.bleu(mtlfin_text,truefin_text),
            "valpred_text": mtlfin_text, 
        }
        return metrics
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        warmup_steps = 2000
        total_steps = steps_per_epoch * self.trainer.max_epochs
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            # cosine decay
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}

    
        



def train(args):
    dmconfig= DataModuleConfig(archive_path=args.archive_path,batch_size=args.batch_size,train_test=args.train_test,train_val=args.train_val,context_len=args.context_len)
    dm = EnFinDataModule(dmconfig)
    dm.setup("fit")
    NUM_HEADS = args.num_heads  
    MODEL_DIM=args.model_dim
    CONTEXT_LEN = args.context_len
    
    decoder_cfg = DecoderConfig(
        num_heads=NUM_HEADS,
        vocab_size=len(dm.train_ds.dataset.tokenizer),
        embedding_size=MODEL_DIM,
        max_seq_len=CONTEXT_LEN,
        atn_cfg=attnconfig(query_dim=MODEL_DIM, key_dim=MODEL_DIM,value_dim= MODEL_DIM,model_dim= MODEL_DIM,
                           n_heads=NUM_HEADS,causal_mask= False,context_len= CONTEXT_LEN,
                           posn_class=PositionalVariant(args.posn_class),posn_weight=args.posn_weight),
        attn_class=AttnVariant.FAST_MULTIHEADED,
        # posn_class=PositionalVariant(1),
        mlp_depth=2,
        post_pre_norm=args.post_pre_norm
    )
    encoder_cfg = EncoderConfig(
        num_heads=4,
        vocab_size=len(dm.train_ds.dataset.tokenizer),
        embedding_size=MODEL_DIM, ## this the dimensions of the output of the encoder
        max_seq_len=CONTEXT_LEN,
        atn_cfg=attnconfig(query_dim=MODEL_DIM,key_dim= MODEL_DIM,value_dim= MODEL_DIM,model_dim= MODEL_DIM,
                           n_heads=NUM_HEADS,causal_mask= False,context_len= CONTEXT_LEN,
                           posn_class=PositionalVariant(args.posn_class),posn_weight=args.posn_weight),
        # pos_weight=0.2,
        mlp_depth=2,
        attn_class=AttnVariant(4),  # Assuming 4 corresponds to a FastMHA variant
        # posn_class=PositionalVariant(1),
        post_pre_norm=args.post_pre_norm
    )
    transformer_cfg = TransformerConfig(
        n_blocks=args.n_blocks,
        encoder_cfg=encoder_cfg,
        decoder_cfg=decoder_cfg,
        tokenizer=dm.train_ds.dataset.tokenizer
    )
    TransformerModel = Transformer(transformer_cfg)
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=True,
        mode='min'
    )
    model_checkpoint = ModelCheckpoint(
        dirpath=args.checkpoint_path,
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )
    
    wandb_logger = WandbLogger(
        project="eng-fin-transformer",
        log_model=False,
        name=f"run-{args.model_dim}d-{args.n_blocks}b-{args.num_heads}h" # Optional: A nice name for the run
    )
    trainer = Trainer(
        accelerator="auto",
        strategy="auto",
        devices=args.devices,
        num_nodes=args.num_nodes,
        logger= wandb_logger,   
        callbacks=[early_stopping,model_checkpoint],
        log_every_n_steps=10,
        
        # track_grad_norm=2,       
    )
    trainer.fit(model=TransformerModel,
                datamodule=dm,
                )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model for Eng-Fin translation.")

    # Data arguments
    parser.add_argument('--archive_path', type=str, required=True, help='Path to the data archive file.')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints', help='Directory to save model checkpoints.')
    
    # Model Hyperparameters
    parser.add_argument('--context_len', type=int, default=512, help='Maximum sequence length for the model.')
    parser.add_argument('--model_dim', type=int, default=512, help='Dimension of the model embeddings.')
    parser.add_argument('--n_blocks', type=int, default=4, help='Number of encoder and decoder blocks.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--posn_class', type=int,default=1,help="1: RoPE, 2: RelativePE")
    parser.add_argument('--posn_weight',type=float,default=0.2,help="Weight of Positional Embedding to QKV matrices")
    parser.add_argument('--post_pre_norm',type=int,default=0,help="Argument to switch between pre and post normalization.")

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size.')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices (GPUs/CPUs) to use.')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes for distributed training.')
    parser.add_argument('--train_test', type=float, default=0.8, help='Proportion of data for the training set.')
    parser.add_argument('--train_val', type=float, default=0.8, help='Proportion of the training set to use for validation.')
    

    args = parser.parse_args()
    train(args)