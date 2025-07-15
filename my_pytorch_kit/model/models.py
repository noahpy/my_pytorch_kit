import torch
import torch.nn as nn
from abc import abstractmethod

class BaseModel(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    @abstractmethod
    def calc_loss(self, batch, criterion) -> torch.Tensor:
        pass

