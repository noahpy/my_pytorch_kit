import torch
import torch.nn as nn
from abc import abstractmethod

class BaseModel(nn.Module):
    """
    Abstract ase class for all models.
    Requires implementation of the calc_loss function.
    Implements weight intialization functions.
    """

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

    def xavier_init(m):
        """
        Xavier initialization for linear layers with ReLU
        """
        if type(m) is nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)
