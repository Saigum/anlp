from encoder import EncoderConfig,TransformerEncoderBlock,TransformerEncoder
from decoder import DecoderConfig,TransformerDecoderBlock,TransformerDecoder
from dataclasses import dataclass
from utils import *
import torch
from torch import Tensor,nn
import lightning
from attention import attnconfig,AttnVariant
from typing import Optional
from lightning import Trainer
import wandb
from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint,RichModelSummary
from lightning.pytorch.loggers import WandbLogger
import argparse 
@dataclass 
class TransformerConfig:
    encoder_cfg:EncoderConfig
    decoder_cfg:DecoderConfig
    tokenizer:GPT2Tokenizer
    n_blocks:int=4

class Transformer(lightning.LightningModule):
    def __init__(self, config:TransformerConfig):
        super().__init__()
        self.encoder =  TransformerEncoder(config.encoder_cfg,config.n_blocks)
        self.decoder =  TransformerDecoder(config.decoder_cfg,config.n_blocks)
        self.tokenizer = config.tokenizer
        self.loss =  torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id,label_smoothing=0.1)
    def forward(self,eng_tokens,fin_tokens,en_mask:Optional[Tensor],fin_mask:Optional[Tensor]):
        encoder_outputs = self.encoder(eng_tokens,en_mask)        
        decoder_output = self.decoder(fin_tokens,encoder_outputs,en_mask,fin_mask)
        return decoder_output
    def training_step(self,batch):
        en_tokens,en_mask,fin_tokens,fin_mask = batch
        probs = self(en_tokens,fin_tokens,en_mask,fin_mask)
        probs = probs[...,:-1,:].reshape(-1,probs.shape[-1]) ## last dim is of size vocab_size, So now (batch x num_tokens), vocab_size tensor. (2-D) tensor
        loss = self.loss(probs,fin_tokens[:,1:].reshape(-1)) ## this will now be batch_size x sequence_length long; (1-D) tensor
    ## Start from token 2 onwards, as i want it to predict these tokens, and drop the last logit, as i dont care for the output given the entire sequence
        self.log("train_loss", loss, prog_bar=True,on_step=True,on_epoch=True)
        return loss
    def validation_step(self,batch):
        en_tokens,en_mask,fin_tokens,fin_mask = batch
        probs = self(en_tokens,fin_tokens,en_mask,fin_mask)  ## to not give it the last token, which i want it to predict.
        probs = probs[...,:-1,:].reshape(-1,probs.shape[-1]) ## last dim is of size vocab_size, So now (batch x num_tokens), vocab_size tensor. (2-D) tensor
        val_loss = self.loss(probs,fin_tokens[:,1:].reshape(-1)) ## this will now be batch_size x sequence_length long; (1-D) tensor
    ## Start from token 2 onwards, as i want it to predict these tokens, and drop the last logit, as i dont care for the output given the entire sequence
        self.log("val_loss",val_loss,prog_bar=True,on_step=True,on_epoch=True)
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
    
    def predict_step(self,batch):
        return super().predict_step()
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
        atn_cfg=attnconfig(MODEL_DIM, MODEL_DIM, MODEL_DIM, MODEL_DIM, NUM_HEADS, False, CONTEXT_LEN),
        attn_class=AttnVariant.FAST_MULTIHEADED,
        posn_class=PositionalVariant(1),
        mlp_depth=2,
    )
    encoder_cfg = EncoderConfig(
        num_heads=4,
        vocab_size=len(dm.train_ds.dataset.tokenizer),
        embedding_size=MODEL_DIM, ## this the dimensions of the output of the encoder
        max_seq_len=CONTEXT_LEN,
        atn_cfg=attnconfig(MODEL_DIM, MODEL_DIM, MODEL_DIM, MODEL_DIM, NUM_HEADS, False, CONTEXT_LEN),
        pos_weight=0.2,
        mlp_depth=2,
        attn_class=AttnVariant(4),  # Assuming 4 corresponds to a FastMHA variant
        posn_class=PositionalVariant(1),
        
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

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size.')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices (GPUs/CPUs) to use.')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes for distributed training.')
    parser.add_argument('--train_test', type=float, default=0.8, help='Proportion of data for the training set.')
    parser.add_argument('--train_val', type=float, default=0.8, help='Proportion of the training set to use for validation.')
    

    args = parser.parse_args()
    train(args)