
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


from my_pytorch_kit.model.models import BaseModel
from my_pytorch_kit.model.utils import ConvArchitect, AffineArchitect


class ImageAE(BaseModel):
    """
    Autoencoder class for images using Convolutional Layers.
    """

    def __init__(self, 
                 *,
                 input_shape = (1, 28, 28),
                 encoder_num_layers = 3,
                 decoder_num_layers = 3,
                 feature_space = (8, 3, 3),
                 latent_dim = 2,
                 **kwargs):
        """
        Initializes the Autoencoder.

        Parameters
        ----------
        input_shape: tuple
            Shape of the input image.
        encoder_num_layers: int
            Number of encoder layers.
        decoder_num_layers: int
            Number of decoder layers.
        feature_space: tuple
            Shape of the output of the convolutional layers.
        latent_dim: int
            Dimension of the latent space.

        """
        super().__init__()

        self.encoder = ImageEncoder(input_shape, encoder_num_layers, latent_dim, feature_space)
        self.decoder = ImageDecoder(input_shape, decoder_num_layers, latent_dim, feature_space)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        x_hat = self.sigmoid(x_hat)
        return x_hat

    def calc_loss(self, batch, criterion):
        x, _ = batch
        x_hat = self.forward(x)
        reconstruction_loss = criterion(x_hat, x)
        return reconstruction_loss


class ImageEncoder(nn.Module):
    """
    Encoder using Convolutional Layers.
    """

    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 encoder_num_layers,
                 latent_dim,
                 conv_output_shape):

        """
        Intializes the Encoder.
        The dimensions of the input is reduced to the conv_output_shape by the convolutional layers.
        This is then finally reduced to latent_dim by a linear layer.

        Parameters
        ----------
        input_shape: tuple
            Shape of the input image.
        encoder_num_layers: int
            Number of encoder layers.
        latent_dim: int
            Dimension of the latent space.
        conv_output_shape: tuple
            Shape of the output of the convolutional layers.
        """

        super().__init__()

        self.architect = ConvArchitect()
        self.conv = self.architect.build(input_shape, conv_output_shape, nn.PReLU(), encoder_num_layers, last_actication=nn.PReLU())
        self.affine = nn.Linear(np.prod(conv_output_shape), latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.affine(x)
        return x


class ImageDecoder(nn.Module):
    """
    Decoder using Convolutional Layers.
    """

    def __init__(self, 
                 output_shape: Tuple[int, int, int],
                 decoder_num_layers: int, 
                 latent_dim: int, 
                 conv_input_shape: Tuple[int, int, int]):
        """
        Initializes the Decoder.
        The latent space is expanded to conv_input_shape by a linear layer.
        This is then finally expanded to output_shape by the convolutional layers.

        Parameters
        ----------
        output_shape: tuple
            Shape of the output image.
        decoder_num_layers: int
            Number of decoder layers.
        latent_dim: int
            Dimension of the latent space.
        conv_input_shape: tuple
            Shape of the input to the convolutional layers.
        """
        super().__init__()

        self.conv_input_shape = conv_input_shape

        self.architect = ConvArchitect()

        self.affine = nn.Linear(latent_dim, np.prod(conv_input_shape))
        self.conf = self.architect.build(conv_input_shape,
                                         output_shape,
                                         nn.PReLU(),
                                         decoder_num_layers,
                                         last_actication=nn.PReLU())

    def forward(self, x):
        x = self.affine(x)
        x = x.view(x.size(0), *self.conv_input_shape)
        x = self.conf(x)
        return x
