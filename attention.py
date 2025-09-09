import os
import torch
from torch import nn
from dataclasses import dataclass
import numpy as np
from typing import Optional,Sequence
from torch import Tensor
from enum import Enum


@dataclass
class attnconfig:
    query_dim:int
    key_dim:int
    value_dim:int
    model_dim:int
    n_heads:int
    causal_mask:bool=False
    context_len:int=512
    
    
class AttnVariant(Enum):
    SLOW_MULTIHEADED=1
    MASKED_SLOW_MULTIHEADED=2
    FAST_MULTIHEADED=3
    FAST_SELFMHA=4

def make_attention(attn_class:AttnVariant,atn_config:attnconfig):
    if attn_class == AttnVariant.SLOW_MULTIHEADED:
        return  MultiHeadedAttention(atn_config)
    elif attn_class == AttnVariant.MASKED_SLOW_MULTIHEADED:
        return MaskedMultiHeadAttention(atn_config)
    elif attn_class == AttnVariant.FAST_MULTIHEADED:
        return FastMHA(config=atn_config)
    elif attn_class == AttnVariant.FAST_SELFMHA:
        return FastSelfAttn(config=atn_config)
    else:
        raise
    
    
class MultiHeadedAttention(nn.Module):
    def __init__(self,config:attnconfig):
        super().__init__()
        self.Wq = nn.ModuleList([nn.Linear(config.query_dim,config.model_dim//config.n_heads) for _ in range(config.n_heads)])
        self.Wk = nn.ModuleList([nn.Linear(config.key_dim,config.model_dim//config.n_heads) for _ in range(config.n_heads)])
        self.Wv = nn.ModuleList([nn.Linear(config.value_dim,config.model_dim//config.n_heads) for _ in range(config.n_heads)])
        self.sf = nn.Softmax(dim=-1)
        self.config=config
    def forward(self,query_vector,key_vector,value_vector,padding_mask:Optional[Tensor]=None):
        output =[]
        for i in range(self.config.n_heads):
            q=self.Wq[i](query_vector)
            k=self.Wk[i](key_vector)
            v=self.Wv[i](value_vector)
            # print(f"Dimensions of k transpose: {k.T.shape}")
            A = self.sf(torch.matmul(q,k.mT)/np.sqrt(q.shape[-1]))
            output.append(A@v)
            # print(f"Shape of head {i} output; {output[-1].shape}")
        return(torch.cat(output,dim=-1))

class MaskedMultiHeadAttention(MultiHeadedAttention):
    def forward(self,query_vector,key_vector,value_vector,padding_mask:Optional[Tensor]=None):
        output =[]
        for i in range(self.config.n_heads):
            q=self.Wq[i](query_vector)
            k=self.Wk[i](key_vector)
            v=self.Wv[i](value_vector)
            A = self.sf(torch.matmul(q,k.mT)/np.sqrt(q.shape[-1]))
            A= torch.tril(input=A)
            output.append(A@v)
            # print(f"Shape of head {i} output; {output[-1].shape}")
        return(torch.cat(output,dim=-1))


class FastMHA(nn.Module):
    '''Class For MultiHeadedAttention, with more paralellized computations.
        Two Main changes:
        1. No for Loop, all attention heads are computed simulataneously.
        2. Q,K,V Computations ar Handled Simultaneously '''
    def __init__(self, config:attnconfig):
        super().__init__( )
        self.W_Q = nn.Linear(config.query_dim,config.model_dim) ## TODO : IT SHOULD BE EMBEDDING_SIZE AS THE FIRST ARGUMENT 
        self.W_K = nn.Linear(config.key_dim,config.model_dim)
        self.W_V = nn.Linear(config.value_dim,config.model_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.sf = nn.Softmax(dim=-1)
        self.config=config
        self.triu_indices = torch.triu_indices(row=config.context_len,col=config.context_len)
        
    def forward(self,query_vector,key_vector,value_vector,padding_mask:Optional[Tensor]=None):
        Qs = self.W_Q(query_vector)
        Ks = self.W_K(key_vector)
        Vs = self.W_V(value_vector)
        n_heads= self.config.n_heads
        Qs = Qs.reshape(Qs.shape[0],n_heads,Qs.shape[1],self.config.model_dim//n_heads)
        Ks = Ks.reshape(Ks.shape[0],n_heads,Ks.shape[1],self.config.model_dim//n_heads)
        Vs = Vs.reshape(Vs.shape[0],n_heads,Vs.shape[1],self.config.model_dim//n_heads)
        As = torch.matmul(Qs,Ks.mT)/np.sqrt(Qs.shape[-1])
        if padding_mask is not None:
            ## padding mask is to be the same shape as the Attention Tensor
            # print(f"Shape of attention matrixL {As.shape}")
            As = As + padding_mask.unsqueeze(1)
        if self.config.causal_mask:
            As[...,self.triu_indices[0],self.triu_indices[1]]  = -1e9
        
        As = self.sf(As)

        # print(f"Shape of attention matrix : {As.shape}")
        ## output shape is batch,n_heads,num_tokens,num_tokens here hopefully ?
        output  = torch.matmul(As,Vs)
        output = self.dropout(output)
        output = output.reshape(output.shape[0],output.shape[2],self.config.model_dim) 
        # print(f"Final output shape : {output.shape}") ## should be batch,n_heads,num_tokens,model_dim
        # print(f"Shape of outputs: {output.shape}")
        if(self.config.causal_mask):
            pass
        return output
    
    ## 384/4
class FastSelfAttn(nn.Module):
    '''Multi-headed Self Attention Module that does the QKV computation in one matmul'''
    def __init__(self, config:attnconfig):
        super().__init__( )
        assert config.query_dim == config.key_dim and config.key_dim == config.value_dim
        self.W_QKV = nn.Linear(config.query_dim,config.model_dim*3)
        self.dropout = nn.Dropout(p=0.2)
        self.sf = nn.Softmax(dim=-1)
        self.config=config
        self.triu_indices = torch.triu_indices(row=config.context_len,col=config.context_len)

    def forward(self,query_vector,key_vector,value_vector,padding_mask:Optional[Tensor]=None):
        ## this is self attention, other args are just dummy args passed in but unused.
        QKV = self.W_QKV(query_vector)
        # print(f"Shape of QKV: {QKV.shape}")
        n_heads = self.config.n_heads
        Qs,Ks,Vs = torch.split(QKV,split_size_or_sections=self.config.model_dim,dim=-1) ## now its batch x num_tokens x (n_heads x model_dim//n_heads)
        # print(f"Current Shape of Qs:{Qs.shape}")
        Qs = Qs.reshape(Qs.shape[0],n_heads,Qs.shape[1],self.config.model_dim//n_heads)
        Ks = Ks.reshape(Ks.shape[0],n_heads,Ks.shape[1],self.config.model_dim//n_heads)
        Vs = Vs.reshape(Vs.shape[0],n_heads,Vs.shape[1],self.config.model_dim//n_heads)
        As = torch.matmul(Qs,Ks.mT)/np.sqrt(Qs.shape[-1])
        if padding_mask is not None:
            # print(f"Shape of padding mask : {padding_mask.shape}")
            # print(f"Shape of Attention Matrix {As.shape}")
            As = As + padding_mask.unsqueeze(1) ## to account for Attention matrix being batch, n_heads, num_tokens, num_tokens
        if( self.config.causal_mask):
            As[...,self.triu_indices[0],self.triu_indices[1]]  = -1e9
        # print(f"Are there Nans in As before softmax {torch.isnan(As).sum()}")
        As = self.sf(As)
        # print(f"This is Attention matrix: {As}")
        
        # print(f"Shape of attention matrix : {As.shape}")
        ## output shape is batch,n_heads,num_tokens,num_tokens here hopefully ?
        output  = torch.matmul(As,Vs)
        output = self.dropout(output)

        output = output.reshape(output.shape[0],output.shape[2],self.config.model_dim) 
        # print(f"Final output shape : {output.shape}") ## should be batch,n_heads,num_tokens,model_dim
        # print(f"Shape of outputs: {output.shape}")
        if(self.config.causal_mask):
            pass
        return output
    
        

class AttentionHead(nn.Module):
    def __init__(self, atn_cfg:attnconfig,embedding_size:int):
        super().__init__()
        self.atn = make_attention(atn_cfg)