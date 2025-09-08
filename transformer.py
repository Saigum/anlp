from decoder import TransformerDecoder
from encoder import TransformerEncoder
import torch
import lightning


class Transformer(lightning.LightningModule):
    def __init__(self, ):
        super().__init__()