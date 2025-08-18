import pytest
import torch
from torch import nn

# Import the class from your file
from my_pytorch_kit.model.utils import ConvArchitect

BATCH_SIZE = 16

@pytest.fixture
def architect():
    """Provides a standard instance of the ConvArchitect for all tests."""
    return ConvArchitect()

def test_encoder_mnist_like(architect):
    """Tests a standard shrinking (encoder) architecture, like for MNIST."""
    num_layers = 3
    input_shape = (1, 28, 28)
    output_shape = (64, 7, 7)

    model = architect.build(input_shape, output_shape, num_layers)
    
    # 1. Check number of layers
    assert len(model) == num_layers
    
    # 2. Check output shape
    dummy_input = torch.randn(BATCH_SIZE, *input_shape)
    output = model(dummy_input)
    assert output.shape == (BATCH_SIZE, *output_shape)

def test_decoder_mnist_like(architect):
    """Tests a standard expanding (decoder) architecture."""
    num_layers = 3
    input_shape = (64, 7, 7)
    output_shape = (1, 28, 28)

    model = architect.build(input_shape, output_shape, num_layers)

    assert len(model) == num_layers
    
    dummy_input = torch.randn(BATCH_SIZE, *input_shape)
    output = model(dummy_input)
    assert output.shape == (BATCH_SIZE, *output_shape)

def test_non_square_encoder(architect):
    """Tests that non-square shapes are handled correctly."""
    num_layers = 2
    input_shape = (3, 32, 64)
    output_shape = (128, 8, 16)

    model = architect.build(input_shape, output_shape, num_layers)
    
    assert len(model) == num_layers
    
    dummy_input = torch.randn(BATCH_SIZE, *input_shape)
    output = model(dummy_input)
    assert output.shape == (BATCH_SIZE, *output_shape)

def test_non_square_decoder(architect):
    """Tests the reverse of the non-square encoder."""
    num_layers = 2
    input_shape = (128, 8, 16)
    output_shape = (3, 32, 64)

    model = architect.build(input_shape, output_shape, num_layers)
    
    assert len(model) == num_layers

    dummy_input = torch.randn(BATCH_SIZE, *input_shape)
    output = model(dummy_input)
    assert output.shape == (BATCH_SIZE, *output_shape)

def test_identity_spatial_transformation(architect):
    """Tests a case where spatial dimensions do not change, only channels."""
    num_layers = 2
    input_shape = (16, 28, 28)
    output_shape = (32, 28, 28)

    model = architect.build(input_shape, output_shape, num_layers)
    
    assert len(model) == num_layers
    
    dummy_input = torch.randn(BATCH_SIZE, *input_shape)
    output = model(dummy_input)
    assert output.shape == (BATCH_SIZE, *output_shape)

def test_single_layer(architect):
    """Tests that the architect works with just one layer."""
    num_layers = 1
    input_shape = (3, 32, 32)
    output_shape = (16, 16, 16)

    model = architect.build(input_shape, output_shape, num_layers)

    assert len(model) == num_layers

    dummy_input = torch.randn(BATCH_SIZE, *input_shape)
    output = model(dummy_input)
    assert output.shape == (BATCH_SIZE, *output_shape)

def test_impossible_transformation_raises_error(architect):
    """
    Tests that the architect correctly raises a ValueError when a
    transformation is not possible with the given constraints.
    """
    num_layers = 1
    # This transformation is very unlikely to be solvable with small kernels
    input_shape = (1, 28, 28)
    output_shape = (64, 1, 1)

    with pytest.raises(ValueError):
        architect.build(input_shape, output_shape, num_layers)
