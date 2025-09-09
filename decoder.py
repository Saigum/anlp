from attention import attnconfig,MultiHeadedAttention,MaskedMultiHeadAttention,FastMHA,FastSelfAttn,make_attention
import torch
from dataclasses import dataclass,field
from torch import nn
from utils import ResMLP
from encoder import make_attention
from typing import Optional
from torch import Tensor
from attention import AttnVariant,PositionalVariant   
import copy
@dataclass 
class DecoderConfig:
    num_heads:int=4
    vocab_size:int=50762
    embedding_size:int=768
    max_seq_len:int=200
    atn_cfg: attnconfig = field(
        default_factory=lambda: attnconfig(
            query_dim=256,
            key_dim=256,
            value_dim=256,
            model_dim=256,
            n_heads=8
        )
    )
    # pos_weight:int=0.2
    mlp_depth:int=1
    attn_class:AttnVariant=AttnVariant.SLOW_MULTIHEADED
    # posn_class:PositionalVariant=PositionalVariant.ROPE
    post_pre_norm:int=1

class TransformerDecoderBlock(nn.Module):
    def __init__(self,config:DecoderConfig):
        super().__init__()
        # self.Embedding = nn.Embedding(config.vocab_size,config.embedding_size)
        # self.PositionalEncoding =  make_positional_embeddings(config.posn_class,config.embedding_size,config.max_seq_len)
        config.atn_cfg.causal_mask = True
        # if config.atn_cfg.model_dim != config.embedding_size:
        #     self.attn_head = nn.Sequential(make_attention(attn_class=config.attn_class,atn_config=config.atn_cfg),
        #                                    nn.Linear(config.atn_cfg.model_dim,config.embedding_size))
        # else:
        self.attn_head = make_attention(attn_class=config.attn_class,atn_config=copy.deepcopy(config.atn_cfg))
        if config.post_pre_norm == 0:
            self.res1 = ResMLP(input_size=config.embedding_size,num_layers=config.mlp_depth)
        else:                          
            self.res1 = nn.Sequential(*[nn.Sequential(nn.Linear(config.embedding_size,config.embedding_size),nn.GELU(),nn.Dropout(p=0.2)) for _ in range(config.mlp_depth)])
        config.atn_cfg.causal_mask = False
        config.atn_cfg.posn_class = PositionalVariant.NONE
        config.atn_cfg.posn_weight = 0
        self.CrossAttention = FastMHA(config=copy.deepcopy(config.atn_cfg))
        
        
        self.layer_norm1 = nn.LayerNorm(config.embedding_size)
        self.layer_norm2 = nn.LayerNorm(config.embedding_size)
        self.layer_norm3 = nn.LayerNorm(config.embedding_size)
        self.decodercfg = config
        
    def forward(self,token_embeddings,encoder_output,source_pad_mask:Optional[Tensor]=None,target_pad_mask:Optional[Tensor]=None):
        # pos_embs = self.PositionalEncoding(token_embeddings)
        # embs = token_embeddings + self.decodercfg.pos_weight*pos_embs
        
        # print(f"Embs shape: {embs.shape}")
        # print(f"This is the pad mask shape: {pad_mask.shape}")
        
        embs = token_embeddings
        
#################### POST NORM VERSION###############################################################
        if self.decodercfg.post_pre_norm == 0:
            embs = self.layer_norm1(self.attn_head(embs,embs,embs,target_pad_mask) + embs) ## target lang pad_mask for 
            # print(f"This is the input to cross attention: ")
            # print(f"This is the encoder_output shape : {encoder_output.shape}")
            embs = self.layer_norm2(embs +  self.CrossAttention(embs,encoder_output,encoder_output,source_pad_mask))
            embs  = self.layer_norm3(self.res1(embs))
################################################################################################################
        
        
###################### PRE NORM VERSION ###############################################################
        else:
            normembs = self.layer_norm1(embs)
            embs  = self.attn_head(normembs,normembs,normembs,target_pad_mask) + embs 
            normembs = self.layer_norm2(embs)
            embs  = self.CrossAttention(normembs,encoder_output,encoder_output,source_pad_mask) + embs
            
            embs  = self.res1(self.layer_norm3(embs)) + embs
####################################################################################################### 
        
        return embs
        
class TransformerDecoder(nn.Module):
    def __init__(self,config:DecoderConfig,n_blocks:int=4):
        super().__init__()
        self.Embedding = nn.Embedding(config.vocab_size,config.embedding_size)
        self.DecoderBlocks = nn.ModuleList([TransformerDecoderBlock(config) for _ in range(n_blocks)])
        self.fc = nn.Sequential(nn.Linear(config.embedding_size,config.vocab_size))
    def forward(self,tokens,encoder_output,source_pad_mask:Optional[Tensor]=None,target_pad_mask:Optional[Tensor]=None):
        embs =  self.Embedding(tokens)
        for i in range(len(self.DecoderBlocks)):
            embs = self.DecoderBlocks[i](embs,encoder_output,source_pad_mask,target_pad_mask)
        probs = self.fc(embs)
        return probs
        
