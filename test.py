import torch
import lightning
from torch import Tensor,nn
import numpy as np
from enum import Enum
from train import Transformer
from dataclasses import dataclass
from typing import Optional
class Strategies(Enum):
    Greedy = 1
    TopK = 2
    Beam = 3

##Dummy Class for Decoding Strategies
class AutoRegressiveGenerator(nn.Module):
    def __init__(self,*args, **kwargs):
        self.tokenizer = kwargs["tokenizer"]
        self.model = Transformer.load_from_checkpoint(checkpoint_path=kwargs["model_checkpoint"])
    def forward(self,logits:Tensor):
        pass
    def predict(self,en_tokens:Tensor,fin_tokens:Tensor):
        with torch.no_grad():
            next_logits = self.model(en_tokens,None,fin_tokens,None)
            logits = self(next_logits)
        return logits
    def generate(self,en_tokens:Tensor,num_steps:int):
        fin_tokens = torch.tensor([self.tokenizer.start_token_id])
        for i in range(num_steps):
            next_fin_token = self.predict(en_tokens=en_tokens,fin_tokens=fin_tokens)
            fin_tokens = torch.cat(tensors=(fin_tokens,torch.tensor(next_fin_token)))
        text = self.tokenizer.decode(fin_tokens,skip_special_tokens=True)
        return text        
class GreedyDecoding(AutoRegressiveGenerator):
    def forward(self, logits):
        return (torch.argmax(logits,dim=-1))
    ### Predicts tokens at next step

class TopKSampling(AutoRegressiveGenerator):
    def __init__(self,*args, **kwargs):
        super().__init__(kwargs["tokenizer"])
        self.k = kwargs["k"]
    def forward(self, logits):
        values,topk = torch.topk(logits,dim=-1)
        topk = topk.detach().cpu().numpy()
        tokens = topk[np.random.randint(low=0,high=self.k,size=(topk.shape[0]))]
        return tokens
    
