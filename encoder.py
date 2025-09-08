from attention import *
import torch
from dataclasses import dataclass
from utils import PositionalEncodings
from torch import nn
from utils import *
from enum import Enum
from tqdm import tqdm
from utils import PositionalVariant
from torch.cuda import device as Device


@dataclass
class EncoderConfig:
      num_heads:int=4
      vocab_size:int=50762
      embedding_size:int=768
      max_seq_len:int=200 ## same thing as num_tokens or context_len
      atn_cfg:attnconfig=attnconfig(query_dim=embedding_size,key_dim=embedding_size,value_dim=embedding_size,model_dim=embedding_size,n_heads=num_heads)
      pos_weight:int=0.2
      mlp_depth:int=1
      attn_class:AttnVariant=AttnVariant.SLOW_MULTIHEADED
      posn_class:PositionalVariant=PositionalVariant.ROPE
      device:Device=torch.device("cpu")

def make_positional_embeddings(posn_class:PositionalVariant,embedding_size:int,max_seq_len:int,device="cpu"):
    if posn_class == PositionalVariant.ROPE:
        return RoPE(embedding_dim=embedding_size,context_len=max_seq_len,device=device)
    elif posn_class == PositionalVariant.RELATIVEPE:
        return RelativePE(embedding_dim=embedding_size,context_len=max_seq_len)
    else:
        raise Exception("Said attention class hasnt been implemented")
    

      

class TransformerEncoderBlock(nn.Module):
    def __init__(self,config:EncoderConfig):
        super().__init__()
        # self.Embedding = nn.Embedding(config.vocab_size,config.embedding_size)
        self.PositionalEncoding =  make_positional_embeddings(config.posn_class,config.embedding_size,config.max_seq_len,config.device)
        
        # if config.atn_cfg.model_dim != config.embedding_size:
        #     self.attn_head = nn.Sequential(make_attention(attn_class=config.attn_class,atn_config=config.atn_cfg),
        #                                    nn.Linear(config.atn_cfg.model_dim,config.embedding_size))
        # else:
        self.attn_head = make_attention(attn_class=config.attn_class,atn_config=config.atn_cfg)
        self.layer_norm1 = nn.LayerNorm(config.embedding_size)
        self.res1 = ResMLP(input_size=config.embedding_size,num_layers=config.mlp_depth)
        self.layer_norm2 = nn.LayerNorm(config.embedding_size)
        self.encodercfg = config
    
    def forward(self,embs,pad_mask:Optional[Tensor]=None):
        # embs = self.Embedding(x)
        pos_embs = self.PositionalEncoding(embs)
        embs = embs + self.encodercfg.pos_weight*pos_embs
        embs = self.layer_norm1(self.attn_head(embs,embs,embs,pad_mask) + embs)
        embs = self.layer_norm2(self.res1(embs))
        return embs
    

class TransformerEncoder(nn.Module):
    def __init__(self,config:EncoderConfig,n_blocks:int=4):
        super().__init__()
        self.Embedding = nn.Embedding(config.vocab_size,config.embedding_size)
        self.EncoderBlocks = nn.ModuleList([TransformerEncoderBlock(config)]*n_blocks)
    def forward(self,tokens,pad_mask:Optional[Tensor]=None):
        embs = self.Embedding(tokens) ## currently token embeddings
        for i in range(len(self.EncoderBlocks)):
            embs  = self.EncoderBlocks[i](embs,pad_mask)
        return embs
            

    
    
        
    
    
    
    
    