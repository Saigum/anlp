import os
import torch
from torch import nn
from dataclasses import dataclass
import numpy as np
from typing import Optional,Sequence
from torch import Tensor
from enum import Enum

    
class PositionalVariant(Enum):
    ROPE=1
    RELATIVEPE=2
    NONE=3


    
## learned positional_encoding, like in GPT-2
class PositionalEncodings(nn.Module):
    def __init__(self,max_seq_len:int,hidden_size:int):
        super().__init__()
        self.pos_emb = nn.Embedding(num_embeddings=max_seq_len,embedding_dim=hidden_size)
    def forward(self,positions):
        ## positions would have to be a set of indices
        return(self.pos_emb(positions))


class RoPE(nn.Module):
    def __init__(self, embedding_dim:int,context_len:int):
        super().__init__( )
        self.d = embedding_dim
        thetas = torch.arange(start=0,end=context_len,step=1,dtype=torch.float).view(-1,1) @torch.pow(1e4,-2*torch.arange(start=0,end=self.d-1,step=2)/self.d).repeat_interleave(2).view(1,-1)
        ## this should be an context_len x d size matrix 
        # print(f"Shape of theta matrix is : {thetas.shape}")
        self.register_buffer('costhetas', torch.cos(thetas))
        self.register_buffer('sinethetas', torch.sin(thetas))
        self.register_buffer('even_idx', torch.arange(start=0, end=self.d, step=2, dtype=torch.long))
        self.register_buffer('odd_idx', torch.arange(start=1, end=self.d, step=2, dtype=torch.long))

    def interswap(self,token_embedding):
        swapped = token_embedding.clone()
        odds =  token_embedding[...,self.odd_idx]
        evens = token_embedding[...,self.even_idx]
        swapped[...,self.odd_idx] =  -1*evens
        swapped[...,self.even_idx] = odds
        return swapped
    
    def forward(self,token_embeddings):
        # print(f"Shape of token Embeddings is {token_embeddings.shape}")
        output = token_embeddings*self.costhetas.unsqueeze(0) + self.interswap(token_embeddings)*self.sinethetas.unsqueeze(0)
        return output


class RelativePE(nn.Module):
    def __init__(self, embedding_dim:int,context_len:int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=2*context_len-1,embedding_dim=embedding_dim)
        
    def forward(self,token_embedding:Tensor):
        pass
    

t

class NoPE(nn.Module):
    def __init__(self):
        super().__init__( )
    def forward(self,x):
        return x
    

def make_positional_embeddings(posn_class:PositionalVariant,embedding_size:int,max_seq_len:int):
    if posn_class == PositionalVariant.ROPE:
        return RoPE(embedding_dim=embedding_size,context_len=max_seq_len)
    elif posn_class == PositionalVariant.RELATIVEPE:
        return RelativePE(embedding_dim=embedding_size,context_len=max_seq_len)
    elif posn_class == PositionalVariant.NONE:
        return NoPE()
    else:
        raise Exception("Said attention class hasnt been implemented")
        
@dataclass
class attnconfig:
    query_dim:int
    key_dim:int
    value_dim:int
    model_dim:int
    n_heads:int
    causal_mask:bool=False
    context_len:int=512
    posn_class:PositionalVariant = PositionalVariant.ROPE
    posn_weight:float=0.2
    
    
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
        self.PositionalEmbeddings = make_positional_embeddings(config.posn_class,
                                                               embedding_size=config.model_dim//config.n_heads,
                                                               max_seq_len=config.context_len)
        ## batch , n_heads , context_length  , model_dim//n_heads
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
       ####positional embeddings#############
        Qs = self.PositionalEmbeddings(Qs)
        Ks = self.PositionalEmbeddings(Ks)
        #################################
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
        self.PositionalEmbeddings = make_positional_embeddings(config.posn_class,
                                                               embedding_size=config.model_dim//config.n_heads,
                                                               max_seq_len=config.context_len)
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
        
        ####positional embeddings#############
        Qs = self.PositionalEmbeddings(Qs)
        Ks = self.PositionalEmbeddings(Ks)
        #################################
        
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
    

class RelativePEMHA(nn.Module):
    def __init__(self, config:attnconfig):
        super().__init__( )
        self.W_Q = nn.Linear(config.query_dim,config.model_dim) ## TODO : IT SHOULD BE EMBEDDING_SIZE AS THE FIRST ARGUMENT 
        self.W_K = nn.Linear(config.key_dim,config.model_dim)
        self.W_V = nn.Linear(config.value_dim,config.model_dim)
        self.R = nn.Embedding(2*config.context_len-1,config.model_dim//config.n_heads ) 
        ## batch , n_heads , context_length  , model_dim//n_heads
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
       ####positional embeddings#############
        Srel = torch.matmul(Qs,self.R[torch.arange(start=-Qs.shape[2],end=)])
        #################################
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
        return outpu        

class AttentionHead(nn.Module):
    def __init__(self, atn_cfg:attnconfig,embedding_size:int):
        super().__init__()
        self.atn = make_attention(atn_cfg)