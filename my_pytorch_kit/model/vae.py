
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from my_pytorch_kit.model.models import BaseModel
from my_pytorch_kit.model.utils import ConvArchitect, AffineArchitect


class ImageVAE(BaseModel):
    """
    Variational Autoencoder class for images using Convolutional Layers.
    """

    def __init__(self, 
                 *,
                 input_shape = (1, 28, 28),
                 encoder_num_layers = 3,
                 decoder_num_layers = 3,
                 latent_dim = 2,
                 feature_space = (8, 3, 3),
                 alpha = 1,
                 beta = 1e-2,
                 **kwargs):
        """
        Initializes the Variational Autoencoder.

        Parameters
        ----------
        input_shape: tuple
            Shape of the input image.
        encoder_num_layers: int
            Number of encoder layers.
        decoder_num_layers: int
            Number of decoder layers.
        latent_dim: int
            Dimension of the latent space.
        feature_space: tuple
            Shape of the output of the convolutional layers.
        beta: float
            Weight of the KL divergence loss.

        """
        super().__init__()

        self.alpha = alpha
        self.beta = beta

        self.encoder = VariationalEncoder(input_shape, encoder_num_layers, latent_dim, feature_space)
        self.decoder = VariationalDecoder(input_shape, decoder_num_layers, latent_dim, feature_space)
        self.sampler = VariationalSampler(latent_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        params = self.encoder(x)
        self.params = params
        z = self.sampler(params)
        x_hat = self.decoder(z)
        x_hat = self.sigmoid(x_hat)
        return x_hat

    def calc_loss(self, batch, criterion):
        x, _ = batch
        x_hat = self.forward(x)
        reconstruction_loss = criterion(x_hat, x)
        kl_div = self.kl_loss(self.params)

        batch_size = x.size(0)
        loss = (self.alpha * reconstruction_loss / batch_size) + (self.beta * kl_div / batch_size)

        return loss

    def kl_loss(self, params):
        """
        Given a batch of encoder output params with shape [batch_size, 2 * latent_dim],
        returns the KL divergence loss.
        """
        mu, log_var = params.chunk(2, dim=-1)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return kl_loss


class ImageVAESemiSupervised(ImageVAE):
    """
    Variational Autoencoder class for images using Convolutional Layers.
    Penalizes latent space overlapping by introducing a classifier besides the decoder.
    """

    def __init__(self, 
                 *,
                 input_shape = (1, 28, 28),
                 encoder_num_layers = 3,
                 decoder_num_layers = 3,
                 latent_dim = 2,
                 feature_space = (8, 3, 3),
                 alpha = 1,
                 beta = 1e-2,
                 num_classes = 10,
                 classifier_num_layers = 2,
                 classifier_activation = nn.PReLU(),
                 classifier_loss_weight = 1e-2,
                 **kwargs):
        super().__init__(
            input_shape = input_shape,
            encoder_num_layers = encoder_num_layers,
            decoder_num_layers = decoder_num_layers,
            latent_dim = latent_dim,
            feature_space = feature_space,
            alpha = alpha,
            beta = beta,
            **kwargs
        )

        self.classifier_num_classes = num_classes
        self.classifier_loss_weight = classifier_loss_weight
        self.classifier = AffineArchitect().build(latent_dim, num_classes, classifier_activation, classifier_num_layers)


    def forward(self, x):
        params = self.encoder(x)
        self.params = params
        z = self.sampler(params)
        x_hat, y_hat = self.generate(z)
        return x_hat, y_hat

    def generate(self, z):
        x_hat = self.decoder(z)
        x_hat = self.sigmoid(x_hat)
        y_hat = self.classifier(z)
        y_hat = self.sigmoid(y_hat)
        return x_hat, y_hat


    def calc_loss(self, batch, criterion):
        x, y = batch
        x_hat, y_hat = self.forward(x)
        reconstruction_loss = criterion(x_hat, x)
        kl_div = self.kl_loss(self.params)

        one_hot = nn.functional.one_hot(y, self.classifier_num_classes).float()
        classifier_loss = criterion(y_hat, one_hot)

        batch_size = x.size(0)
        loss = (self.alpha * reconstruction_loss / batch_size) + (self.beta * kl_div / batch_size) + (self.classifier_loss_weight * classifier_loss / batch_size)

        return loss



class VariationalEncoder(nn.Module):
    """
    Variational Encoder using Convolutional Layers.
    """

    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 encoder_num_layers: int,
                 latent_dim: int,
                 conv_output_shape: Tuple[int, int, int],
                 activation: nn.Module = nn.PReLU()):

        """
        Intializes the Variational Encoder.
        The dimensions of the input is reduced to the conv_output_shape by the convolutional layers.
        This is then finally reduced to 2 * latent_dim by a linear layer.

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
        activation: nn.Module
            Activation function for the convolutional layers.
        """

        super().__init__()

        self.architect = ConvArchitect()
        self.conv = self.architect.build(input_shape, conv_output_shape, activation, encoder_num_layers, last_actication=activation)
        self.affine = nn.Linear(np.prod(conv_output_shape), latent_dim * 2)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.affine(x)
        return x


class VariationalDecoder(nn.Module):
    """
    Variational Decoder using Convolutional Layers.
    """

    def __init__(self, 
                 output_shape: Tuple[int, int, int],
                 decoder_num_layers: int, 
                 latent_dim: int, 
                 conv_input_shape: Tuple[int, int, int],
                 activation: nn.Module = nn.PReLU()):
        """
        Initializes the Variational Decoder.
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
        self.conf = self.architect.build(conv_input_shape, output_shape, activation, decoder_num_layers)

    def forward(self, x):
        x = self.affine(x)
        x = x.view(x.size(0), *self.conv_input_shape)
        x = self.conf(x)
        return x


class VariationalSampler(nn.Module):
    """
    Sample layer class between the encoder and decoder.

    Given the mean and log variance of the latent space, this layer samples from a normal distribution.
    """

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, x):
        """
        Input shall be of shape [batch_size, 2 * latent_dim].
        """
        mean, log_var = x.chunk(2, dim=-1)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
