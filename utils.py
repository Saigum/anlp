import os
from torch.utils.data import random_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import polars as po
import numpy as np
import pytorch_lightning as pl
import lightning
from transformers import GPT2Tokenizer
from dataclasses import dataclass
from torch.utils.data import DataChunk
from enum import Enum
from torch.utils.data import random_split
from torch import Generator


class EnFinnishDataset(torch.utils.data.Dataset):
    def __init__(self,archive_path:str,context_len:int=512):
        super().__init__()
        with open(os.path.join(archive_path,"EUbookshop.en")) as fp:
            self.english_corpus = fp.readlines()
        with open(os.path.join(archive_path,"EUbookshop.fi")) as fp:
            self.finnish_corpus = fp.readlines()
        self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        self.tokenizer.add_special_tokens({'pad_token': '<pad>',"bos_token": "<bos>"})        
        print("PAD token:", self.tokenizer.pad_token, self.tokenizer.pad_token_id)
        print("EOS token:", self.tokenizer.eos_token, self.tokenizer.eos_token_id)
        print("BOS token:", self.tokenizer.bos_token, self.tokenizer.bos_token_id)
        self.context_len = context_len
        print(self.tokenizer.pad_token)      
        print(self.tokenizer.pad_token_id)     

    def return_masks(self,pad_idx:int):
        pad_masks = torch.zeros(size=(self.context_len,self.context_len))
        pad_masks[:,pad_idx:] = -1e9
        pad_masks[pad_idx:,:] = -1e9
        return pad_masks
    def __getitem__(self, index):
        en_tokens = torch.tensor(self.tokenizer(self.english_corpus[index],padding="max_length",max_length=self.context_len,truncation=True)["input_ids"])
        finnish_tokens = torch.tensor(self.tokenizer(self.finnish_corpus[index],padding="max_length",max_length=self.context_len,truncation=True)["input_ids"])
        en_pad_indices = torch.where(en_tokens==self.tokenizer.pad_token_id)[0]
        en_pad_index = en_pad_indices[0] if len(en_pad_indices) >0 else self.context_len
        fin_pad_indices = torch.where(finnish_tokens == self.tokenizer.pad_token_id)[0]
        fin_pad_index = fin_pad_indices[0] if len(fin_pad_indices) >0 else self.context_len
        en_pad_masks = self.return_masks(en_pad_index)
        fin_pad_masks = self.return_masks(fin_pad_index)
        # en_pad_masks = torch.concat([torch.full((en_pad_index,),fill_value=0),torch.full((self.context_len-en_pad_index,),-torch.inf)])
        # fin_pad_masks = torch.concat([torch.full((fin_pad_index,),fill_value=0),torch.full((self.context_len-fin_pad_index,),-1*torch.inf)])
        # en_pad_masks = en_pad_masks.view(-1,1)@torch.concat([torch.zeros((fin_pad_index,)),torch.ones(self.context_len-fin_pad_index)])
        # print(en_pad_masks)
        return (en_tokens,en_pad_masks,finnish_tokens,fin_pad_masks)
    def __len__(self):
        return len(self.english_corpus)


@dataclass
class DataModuleConfig:
    archive_path:str="EUbookshop-1"
    batch_size:int=32
    train_test:float=0.8
    train_val:float=0.8 
    context_len:int=512
     
class EnFinDataModule(lightning.LightningDataModule):
    def __init__(self,
                 config:DataModuleConfig):
        super().__init__()
        self.config = config

    def setup(self, stage: str):
        if not hasattr(self, 'FullDataset'):
            self.FullDataset = EnFinnishDataset(self.config.archive_path,context_len=self.config.context_len)
        if not hasattr(self, 'train_ds'):
            full_len = len(self.FullDataset)
            train_len = int(self.config.train_test * full_len)
            test_len = full_len - train_len
            val_len = int((1-self.config.train_val) * train_len)
            train_len = train_len - val_len
            print(f"Dataset lengths: Train={train_len}, Val={val_len}, Test={test_len}")
            generator = Generator().manual_seed(42)
            self.train_ds, self.val_ds, self.test_ds = random_split(
                self.FullDataset,
                [train_len, val_len, test_len],
                generator=generator
            )
    def train_dataloader(self):
        return DataLoader(self.train_ds,batch_size=self.config.batch_size)
    def val_dataloader(self):
        return DataLoader(self.val_ds,batch_size=self.config.batch_size)
    def test_dataloader(self):
        return DataLoader(self.test_ds,batch_size=self.config.batch_size)
    
####################################################################################
################    POSITIONAL EMBEDDINGS  #########################################
####################################################################################


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
        thetas = torch.arange(start=0,end=context_len,step=1,dtype=torch.float).view(-1,1) @torch.pow(1e5,-2*torch.arange(start=0,end=self.d-1,step=2)/self.d).repeat_interleave(2).view(1,-1)
        ## this should be an context_len x d size matrix 
        # print(f"Shape of theta matrix is : {thetas.shape}")
        self.register_buffer('costhetas', torch.cos(thetas))
        self.register_buffer('sinethetas', torch.sin(thetas))
        self.register_buffer('even_idx', torch.arange(start=0, end=self.d, step=2, dtype=torch.long))
        self.register_buffer('odd_idx', torch.arange(start=1, end=self.d, step=2, dtype=torch.long))

    def interswap(self,token_embedding):
        odds =  token_embedding[...,self.odd_idx]
        evens = token_embedding[...,self.even_idx]
        token_embedding[...,self.odd_idx] =  -1*evens
        token_embedding[...,self.even_idx] = odds
        return token_embedding
    
    def forward(self,token_embeddings):
        # print(f"Shape of token Embeddings is {token_embeddings.shape}")
        output = token_embeddings*self.costhetas.unsqueeze(0) + self.interswap(token_embeddings)*self.sinethetas.unsqueeze(0)
        return output


class RelativePE(nn.Module):
    def __init__(self, embedding_dim:int,context_len:int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=context_len,embedding_dim=embedding_dim)
        
        



class ResMLP(nn.Module):
    def __init__(self, input_size:int,num_layers:int):
        super().__init__()
        self.Linears = nn.ModuleList([nn.Sequential(nn.Linear(input_size,input_size),nn.GELU()) for _ in range(num_layers)])
    def forward(self,x):
        res =x
        for i in range(len(self.Linears)):
            x = self.Linears[i](x)  
        return res+x


    
class PositionalVariant(Enum):
    ROPE=1
    RELATIVEPE=2
