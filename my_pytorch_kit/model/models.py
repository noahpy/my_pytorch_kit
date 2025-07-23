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


    def kaiming_init(m):
        """
        Kaiming initialization for linear layers with ReLU
        """
        if type(m) is nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.01)
