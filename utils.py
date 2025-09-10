import os
from torch.utils.data import random_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import polars as po
import numpy as np
import pytorch_lightning as pl
import lightning
from transformers import GPT2Tokenizer,AutoTokenizer
from dataclasses import dataclass
from torch.utils.data import DataChunk
from enum import Enum
from torch.utils.data import random_split
from torch import Generator

    
def _first_pad_index(ids: torch.Tensor, pad_id: int, L: int) -> int:
    idx = (ids == pad_id).nonzero(as_tuple=True)[0]
    return int(idx[0].item()) if idx.numel() > 0 else L

def _pad_bool_matrix(pad_idx: int, L: int) -> torch.Tensor:
    m = torch.zeros((L, L), dtype=torch.bool)
    if pad_idx < L:
        m[:, pad_idx:] = True
        m[pad_idx:, :] = True
    return m

class EnFinnishDataset(torch.utils.data.Dataset):
    def __init__(self,archive_path:str,context_len:int=512,adding_padding:int=0,keys_only:bool=True):
        super().__init__()
        with open(os.path.join(archive_path,"EUbookshop.en")) as fp:
            self.english_corpus = fp.readlines()
        with open(os.path.join(archive_path,"EUbookshop.fi")) as fp:
            self.finnish_corpus = fp.readlines()
        
        # self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fi")
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        print(self.tokenizer.special_tokens_map)
        # self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        # self.tokenizer.add_special_tokens({"bos_token": "<bos>"})        
        # print("PAD token:", self.tokenizer.pad_token, self.tokenizer.pad_token_id)
        # print("EOS token:", self.tokenizer.eos_token, self.tokenizer.eos_token_id)
        # print("BOS token:", self.tokenizer.bos_token, self.tokenizer.bos_token_id)
        self.context_len = context_len
        print(self.tokenizer.pad_token)      
        print(self.tokenizer.pad_token_id)    
        self.adding_padding=adding_padding
        self.keys_only  = keys_only

    def return_masks(self,pad_idx:int,keys_only:bool):
        pad_masks = torch.zeros(size=(self.context_len,self.context_len))
        pad_masks[:,pad_idx:] = -1e9
        if not keys_only:
            pad_masks[pad_idx:,:] = -1e9
        return pad_masks
    def __getitem__(self, index):
        en_tokens = torch.tensor(self.tokenizer(self.english_corpus[index],padding="max_length",max_length=self.context_len,truncation=True)["input_ids"])
        finnish_tokens = torch.tensor([self.tokenizer.bos_token_id] + self.tokenizer(self.finnish_corpus[index],padding="max_length",max_length=self.context_len-1,truncation=True)["input_ids"])
        ## for teacher forcing 
        en_pad_indices = torch.where(en_tokens==self.tokenizer.pad_token_id)[0]
        en_pad_index = en_pad_indices[0] if len(en_pad_indices) >0 else self.context_len
        fin_pad_indices = torch.where(finnish_tokens == self.tokenizer.pad_token_id)[0]
        fin_pad_index = fin_pad_indices[0] if len(fin_pad_indices) >0 else self.context_len
        if self.adding_padding:
            en_pad_masks = self.return_masks(en_pad_index,keys_only=self.keys_only)
            fin_pad_masks = self.return_masks(fin_pad_index,keys_only=self.keys_only)
        else:
            en_pad_masks  = _pad_bool_matrix(en_pad_index,self.context_len)
            fin_pad_masks = self.return_masks(fin_pad_index,self.context_len)
        # en_pad_masks = torch.concat([torch.full((en_pad_index,),fill_value=0),torch.full((self.context_len-en_pad_index,),-torch.inf)])
        # fin_pad_masks = torch.concat([torch.full((fin_pad_index,),fill_value=0),torch.full((self.context_len-fin_pad_index,),-1*torch.inf)])
        # en_pad_masks = en_pad_masks.view(-1,1)@torch.concat([torch.zeros((fin_pad_index,)),torch.ones(self.context_len-fin_pad_index)])
        # print(en_pad_masks)
        return (en_tokens,en_pad_masks,finnish_tokens,fin_pad_masks)
    def __len__(self):
        return len(self.english_corpus)

import os
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from collections import Counter
import random
import math
import pickle as pk




def _seq_is_repetitive(ids: List[int], max_token_ratio: float = 0.6,
                       unique_ratio_floor: float = 0.3,
                       max_consecutive_dupes: int = 4) -> bool:
    if not ids:
        return True
    L = len(ids)
    if L < 3:
        return False

    cnt = Counter(ids)
    if max(cnt.values()) / L >= max_token_ratio:
        return True
    if (len(cnt) / L) <= unique_ratio_floor:
        return True

    cons = 1
    for a, b in zip(ids[:-1], ids[1:]):
        if a == b:
            cons += 1
            if cons >= max_consecutive_dupes:
                return True
        else:
            cons = 1
    return False

class CleanedEnFinnishDataset(Dataset):
    def __init__(
        self,
        archive_path: str,
        context_len: int = 512,
        adding_padding:int=0,
        k_sigma: float = 1.0,                 # range: [mu - kσ, mu + kσ]
        remove_repetitions: bool = True,
        dedup: bool = True,
        split: Optional[str] = None,          # None, "train", "val", "test"
        splits: Tuple[float, float, float] = (0.90, 0.05, 0.05),
        seed: int = 42,
        keys_only:int=1
    ):
        super().__init__()
        self.context_len = context_len
        self.adding_padding = adding_padding
        self.keys_only = keys_only
        
        if (os.path.exists(os.path.join(archive_path, "en_tokens.pk")) and 
            os.path.exists(os.path.join(archive_path, "fi_tokens.pk"))):
            with open(os.path.join(archive_path, "en_tokens.pk"), "rb") as f:
                self.en_ids = pk.load(f)
            with open(os.path.join(archive_path, "fi_tokens.pk"), "rb") as f:
                self.fi_ids = pk.load(f)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Helsinki-NLP/opus-mt-en-fi"
            )
            assert self.tokenizer.pad_token_id is not None, "Tokenizer needs a pad_token_id."
            PAD = self.tokenizer.pad_token_id
            print("special_tokens_map:", self.tokenizer.special_tokens_map)
            print("BOS/EOS/PAD:", self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id)
            print(f"[Cleaned] kept {len(self.en_ids)} sentence pairs (split={split})")
            return
        
        # --- Load raw lines ---
        with open(os.path.join(archive_path, "EUbookshop.en")) as fp:
            en_lines = [l.strip() for l in fp.readlines()]
        with open(os.path.join(archive_path, "EUbookshop.fi")) as fp:
            fi_lines = [l.strip() for l in fp.readlines()]

        N = min(len(en_lines), len(fi_lines))
        en_lines, fi_lines = en_lines[:N], fi_lines[:N]
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     "Helsinki-NLP/opus-mt-en-fi"
        # )
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        
        assert self.tokenizer.pad_token_id is not None, "Tokenizer needs a pad_token_id."
        PAD = self.tokenizer.pad_token_id
        def tok_len(txt: str) -> int:
            ids = self.tokenizer(
                txt, padding="max_length", truncation=True, max_length=context_len
            )["input_ids"]
            return sum(1 for t in ids if t != PAD)

        en_lens = [tok_len(s) for s in en_lines]
        fi_lens = [tok_len(s) for s in fi_lines]
        if dedup:
            seen = set()
            keep_idx = []
            for i, (e, f) in enumerate(zip(en_lines, fi_lines)):
                key = (e, f)
                if key not in seen:
                    seen.add(key)
                    keep_idx.append(i)
            en_lines = [en_lines[i] for i in keep_idx]
            fi_lines = [fi_lines[i] for i in keep_idx]
            en_lens = [en_lens[i] for i in keep_idx]
            fi_lens = [fi_lens[i] for i in keep_idx]

        def band_mask(lens: List[int]) -> Tuple[float, float, List[bool]]:
            mu = float(sum(lens) / max(1, len(lens)))
            var = float(sum((x - mu) ** 2 for x in lens) / max(1, len(lens)))
            sigma = math.sqrt(max(0.0, var))
            lo, hi = mu - k_sigma * sigma, mu + k_sigma * sigma
            mask = [(lo <= x <= hi) for x in lens]
            return lo, hi, mask

        en_lo, en_hi, en_ok = band_mask(en_lens)
        fi_lo, fi_hi, fi_ok = band_mask(fi_lens)
        keep_idx = [i for i in range(len(en_lines)) if en_ok[i] and fi_ok[i]]
        en_lines = [en_lines[i] for i in keep_idx]
        fi_lines = [fi_lines[i] for i in keep_idx]
        en_lens = [en_lens[i] for i in keep_idx]
        fi_lens = [fi_lens[i] for i in keep_idx]
        if remove_repetitions:
            keep = []
            for e, f in zip(en_lines, fi_lines):
                e_ids = self.tokenizer(
                    e, padding="max_length", truncation=True, max_length=context_len
                )["input_ids"]
                f_ids = self.tokenizer(
                    f, padding="max_length", truncation=True, max_length=context_len
                )["input_ids"]
                # drop PAD for repetition checks
                e_core = [t for t in e_ids if t != PAD]
                f_core = [t for t in f_ids if t != PAD]
                if not _seq_is_repetitive(e_core) and not _seq_is_repetitive(f_core):
                    keep.append((e, f))
            en_lines, fi_lines = zip(*keep) if keep else ([], [])
            en_lines, fi_lines = list(en_lines), list(fi_lines)
        if split is not None:
            assert split in {"train", "val", "test"}
            rng = random.Random(seed)
            idxs = list(range(len(en_lines)))
            rng.shuffle(idxs)
            n = len(idxs)
            n_train = int(splits[0] * n)
            n_val = int(splits[1] * n)
            train_idx = idxs[:n_train]
            val_idx = idxs[n_train:n_train + n_val]
            test_idx = idxs[n_train + n_val:]

            pick = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
            en_lines = [en_lines[i] for i in pick]
            fi_lines = [fi_lines[i] for i in pick]
            
        self.en_ids = []
        self.fi_ids = []
        for e, f in zip(en_lines, fi_lines):
            e_ids = self.tokenizer(
                e, padding="max_length", truncation=True, max_length=context_len
            )["input_ids"]
            f_ids = self.tokenizer(
                f, padding="max_length", truncation=True, max_length=context_len
            )["input_ids"]
            self.en_ids.append(torch.tensor(e_ids, dtype=torch.long))
            self.fi_ids.append(torch.tensor(f_ids, dtype=torch.long))
            
        with open(os.path.join(archive_path,"en_tokens.pk"),"wb") as fp:
            pk.dump(self.en_ids,fp)
        with open(os.path.join(archive_path,"fi_tokens.pk"),"wb") as fp:
            pk.dump(self.fi_ids,fp)
            
        print("special_tokens_map:", self.tokenizer.special_tokens_map)
        print("BOS/EOS/PAD:", self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id)
        print(f"[Cleaned] kept {len(self.en_ids)} sentence pairs (split={split})")

    def __len__(self):
        return len(self.en_ids)

    def return_masks(self,pad_idx:int,keys_only:bool):
        pad_masks = torch.zeros(size=(self.context_len,self.context_len))
        pad_masks[:,pad_idx:] = -1e9
        if not keys_only:
            pad_masks[pad_idx:,:] = -1e9
        return pad_masks
    
    def __getitem__(self, index):
        en = self.en_ids[index]
        fi = self.fi_ids[index]
        PAD = self.tokenizer.pad_token_id
        L = self.context_len

        en_pad_idx = _first_pad_index(en, PAD, L)
        fi_pad_idx = _first_pad_index(fi, PAD, L)

        if(self.adding_padding):
            en_mask_bool = self.return_masks(en_pad_idx,self.keys_only) ## just naming it this lol coz why not.
            fi_mask_bool = self.return_masks(fi_pad_idx,self.keys_only)
        else:
            en_mask_bool = _pad_bool_matrix(en_pad_idx, L)  
            fi_mask_bool = _pad_bool_matrix(fi_pad_idx, L)

        return en, en_mask_bool, fi, fi_mask_bool


@dataclass
class DataModuleConfig:
    archive_path:str="EUbookshop-1"
    batch_size:int=32
    train_test:float=0.8
    train_val:float=0.8 
    context_len:int=512
    num_workers:int=8
    clean:int=0
    adding_padding:int=0
    keys_only:int =1
     
class EnFinDataModule(lightning.LightningDataModule):
    def __init__(self,
                 config:DataModuleConfig):
        super().__init__()
        self.config = config

    def setup(self, stage: str):
        if not hasattr(self, 'FullDataset'):
            if(self.config.clean):
                self.FullDataset = CleanedEnFinnishDataset(self.config.archive_path,self.config.context_len,self.config.adding_padding,
                                                           keys_only=self.config.keys_only)
            else:    
                self.FullDataset = EnFinnishDataset(self.config.archive_path,context_len=self.config.context_len,adding_padding=self.config.adding_padding,
                                                    keys_only=self.config.keys_only)
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
        return DataLoader(self.train_ds,batch_size=self.config.batch_size,num_workers=8)
    def val_dataloader(self):
        return DataLoader(self.val_ds,batch_size=self.config.batch_size,num_workers=8)
    def test_dataloader(self):
        return DataLoader(self.test_ds,batch_size=self.config.batch_size,num_workers=8)
    
####################################################################################
################    POSITIONAL EMBEDDINGS  #########################################
####################################################################################




class ResMLP(nn.Module):
    def __init__(self, input_size:int,num_layers:int):
        super().__init__()
        self.Linears = nn.ModuleList([nn.Sequential(nn.Linear(input_size,input_size),nn.GELU(),nn.Dropout(p=0.1)) for _ in range(num_layers)])
    def forward(self,x):
        res =x
        for i in range(len(self.Linears)):
            x = self.Linears[i](x)  
        return res+x



