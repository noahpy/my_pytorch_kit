
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


from my_pytorch_kit.model.models import BaseModel
from my_pytorch_kit.model.utils import ConvArchitect, AffineArchitect

class ImageClassifier(BaseModel):
    """
    Image Classifier using Convolutional and Affine Layers.
    """


    def __init__(self, 
                 *,
                 input_shape: Tuple[int, int, int] = (1, 28, 28),
                 conv_num_layers: int = 3,
                 affine_num_layers: int = 2,
                 num_classes: int = 10,
                 feature_space: Tuple[int, int, int] = (32, 7, 7),
                 conv_activation: nn.Module = nn.PReLU(),
                 affine_activation: nn.Module = nn.PReLU(),
                 use_softmax: bool = False,
                 **kwargs):
        """
        Initializes the ImageClassifier.

        Parameters
        ----------
        input_shape: tuple
            Shape of the input image.
        conv_num_layers: int
            Number of convolutional layers.
        affine_num_layers: int
            Number of affine layers.
        num_classes: int
            Number of classes.
        feature_space: tuple
            Shape of the feature space.
        conv_activation: nn.Module
            Activation function for the convolutional layers.
        affine_activation: nn.Module
            Activation function for the affine layers.
        use_softmax: bool
            Whether to use softmax activation function.
        """

        super().__init__()

        self.conv_architect = ConvArchitect()
        self.affine_architect = AffineArchitect()

        self.conv = self.conv_architect.build(input_shape, feature_space, conv_activation, conv_num_layers)
        self.affine = self.affine_architect.build(np.prod(feature_space), num_classes, affine_activation, affine_num_layers)

        self.use_softmax = use_softmax
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.affine(x)
        if self.use_softmax:
            x = self.softmax(x)
        return x

    def calc_loss(self, batch, criterion):
        x, y = batch
        x_hat = self.forward(x)
        return criterion(x_hat, y)
