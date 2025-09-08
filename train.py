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



@dataclass 
class TransformerConfig:
    n_blocks:int=4
    encoder_cfg:EncoderConfig
    decoder_cfg:DecoderConfig
    tokenizer:GPT2Tokenizer

class Transformer(lightning.LightningModule):
    def __init__(self, config:TransformerConfig):
        self.encoder =  TransformerEncoder(config.encoder_cfg,config.n_blocks)
        self.decoder =  TransformerDecoder(config.decoder_cfg,config.n_blocks)
        self.tokenizer = config.tokenizer
        self.loss =  torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)
    def forward(self,eng_tokens,fin_tokens,en_mask:Optional[Tensor],fin_mask:Optional[Tensor]):
        encoder_outputs = self.encoder(eng_tokens,en_mask)        
        decoder_output = self.decoder(fin_tokens,encoder_outputs,fin_mask)
        return decoder_output
    def training_step(self,batch):
        en_tokens,en_mask,fin_tokens,fin_mask = batch
        probs = self(en_tokens,fin_tokens,en_mask,fin_mask)
        loss = self.loss(probs,fin_tokens)
        return loss
    def validation_step(self,batch):
        en_tokens,en_mask,fin_tokens,fin_mask = batch
        probs = self(en_tokens,fin_tokens,en_mask,fin_mask)
        val_loss = self.loss(probs,fin_tokens)
        ## sample some of the words
        return val_loss
    def predict_step(self,batch):
        return super().predict_step()
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
        



def train(args):
    dmconfig= DataModuleConfig(archive_path=args.archive_path,batch_size=args.batch_size,train_test=args.train_test,train_val=args.train_val,context_len=args.context_len)
    dm = EnFinDataModule(dmconfig)
    dm.setup()
    NUM_HEADS = args.num_heads
    MODEL_DIM=args.model_dim
    CONTEXT_LEN = args.context_len
    
    decoder_cfg = DecoderConfig(
        num_heads=NUM_HEADS,
        vocab_size=len(dm.train_ds.tokenizer),
        embedding_size=MODEL_DIM,
        max_seq_len=CONTEXT_LEN,
        atn_cfg=attnconfig(MODEL_DIM, MODEL_DIM, MODEL_DIM, MODEL_DIM, NUM_HEADS, False, CONTEXT_LEN),
        attn_class=AttnVariant.FAST_MULTIHEADED,
        posn_class=PositionalVariant(1),
    )
    encoder_cfg = EncoderConfig(
        num_heads=4,
        vocab_size=len(dm.train_ds.tokenizer),
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
        tokenizer=dm.train_ds.tokenizer
    )
    TransformerModel = Transformer(transformer_cfg)
    
    
    trainer = Trainer(
        accelerator="auto",
        strategy="auto",
        devices=args.devices,
        num_nodes=args.num_nodes,
        logger=       
    )
    
    