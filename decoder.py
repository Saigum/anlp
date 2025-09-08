from attention import attnconfig,MultiHeadedAttention,MaskedMultiHeadAttention,FastMHA,FastSelfAttn,make_attention
import torch
from dataclasses import dataclass
from utils import PositionalEncodings
from torch import nn
from utils import ResMLP
from encoder import make_attention,make_positional_embeddings
from utils import PositionalVariant
from typing import Optional
from torch import Tensor
from attention import AttnVariant
from torch.cuda import device as Device
@dataclass 
class DecoderConfig:
    num_heads:int=4
    vocab_size:int=50762
    embedding_size:int=768
    max_seq_len:int=200
    atn_cfg:attnconfig=attnconfig(query_dim=embedding_size,key_dim=embedding_size,value_dim=embedding_size,model_dim=embedding_size,n_heads=num_heads)
    pos_weight:int=0.2
    mlp_depth:int=1
    attn_class:AttnVariant=AttnVariant.SLOW_MULTIHEADED
    posn_class:PositionalVariant=PositionalVariant.ROPE
    device:Device=torch.device("cpu")
        
class TransformerDecoderBlock(nn.Module):
    def __init__(self,config:DecoderConfig):
        super().__init__()
        # self.Embedding = nn.Embedding(config.vocab_size,config.embedding_size)
        self.PositionalEncoding =  make_positional_embeddings(config.posn_class,config.embedding_size,config.max_seq_len,config.device)
        config.atn_cfg.causal_mask = True
        # if config.atn_cfg.model_dim != config.embedding_size:
        #     self.attn_head = nn.Sequential(make_attention(attn_class=config.attn_class,atn_config=config.atn_cfg),
        #                                    nn.Linear(config.atn_cfg.model_dim,config.embedding_size))
        # else:
        self.attn_head = make_attention(attn_class=config.attn_class,atn_config=config.atn_cfg)
        self.res1 = nn.Sequential( ResMLP(input_size=config.embedding_size,num_layers=config.mlp_depth),
                                  nn.LayerNorm(config.embedding_size))
        config.atn_cfg.causal_mask = False
        self.CrossAttention = FastMHA(config=config.atn_cfg)
        
        
        self.layer_norm1 = nn.LayerNorm(config.embedding_size)
        self.layer_norm2 = nn.LayerNorm(config.embedding_size)
        self.layer_norm3 = nn.LayerNorm(config.embedding_size)
        self.decodercfg = config
        
    def forward(self,token_embeddings,encoder_output,pad_mask:Optional[Tensor]=None):
        pos_embs = self.PositionalEncoding(token_embeddings)
        embs = token_embeddings + self.decodercfg.pos_weight*pos_embs
        
        # print(f"Embs shape: {embs.shape}")
        # print(f"This is the pad mask shape: {pad_mask.shape}")
        embs = self.layer_norm1(self.attn_head(embs,embs,embs,pad_mask) + embs)
        # print(f"This is the input to cross attention: ")
        # print(f"This is the encoder_output shape : {encoder_output.shape}")
        embs = self.layer_norm2(embs +  self.CrossAttention(embs,encoder_output,encoder_output,pad_mask))
        embs  = self.layer_norm3(self.res1(embs))
        return embs
        
class TransformerDecoder(nn.Module):
    def __init__(self,config:DecoderConfig,n_blocks:int=4):
        super().__init__()
        self.Embedding = nn.Embedding(config.vocab_size,config.embedding_size)
        self.DecoderBlocks = nn.ModuleList([TransformerDecoderBlock(config)]*n_blocks)
        self.fc = nn.Sequential(nn.Linear(config.embedding_size,config.vocab_size),
                                nn.Softmax(-1))
    def forward(self,tokens,encoder_output,pad_mask:Optional[Tensor]=None):
        embs =  self.Embedding(tokens)
        for i in range(len(self.DecoderBlocks)):
            embs = self.DecoderBlocks[i](embs,encoder_output,pad_mask)
        probs = self.fc(embs)
        return probs
        
