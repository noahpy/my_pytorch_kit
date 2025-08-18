import pytest
import torch
from torch import nn
import numpy as np
from typing import Tuple, List

# Assume the classes AffineArchitect and ConvArchitect are in a file named `architect.py`
from my_pytorch_kit.model.utils import AffineArchitect, ConvArchitect

# --- Fixtures ---

@pytest.fixture
def affine_architect():
    """Provides a default AffineArchitect instance."""
    return AffineArchitect()

@pytest.fixture
def conv_architect():
    """Provides a default ConvArchitect instance."""
    return ConvArchitect()

# --- Tests for AffineArchitect ---

class TestAffineArchitect:
    """Grouped tests for the AffineArchitect class."""

    def test_identity_case(self, affine_architect):
        """Test that nn.Identity is returned for same-sized input/output."""
        model = affine_architect.build(64, 64, nn.ReLU(), 3)
        assert isinstance(model, nn.Identity)
        test_tensor = torch.randn(10, 64)
        assert torch.equal(model(test_tensor), test_tensor)

    def test_single_layer(self, affine_architect):
        """Test building a single-layer affine network."""
        model = affine_architect.build(128, 16, nn.ReLU(), 1)
        assert isinstance(model, nn.Sequential)
        # A single layer has no intermediate activation
        assert len(model) == 1
        assert isinstance(model[0], nn.Linear)
        assert model[0].in_features == 128
        assert model[0].out_features == 16

    def test_single_layer_with_last_activation(self, affine_architect):
        """Test a single-layer network with a last activation."""
        model = affine_architect.build(128, 16, nn.ReLU(), 1, nn.Sigmoid())
        assert len(model) == 2
        assert isinstance(model[0], nn.Linear)
        assert isinstance(model[1], nn.Sigmoid)

    @pytest.mark.parametrize("input_size, output_size, num_layers", [
        (256, 16, 4),  # Decreasing
        (16, 256, 4),  # Increasing
        (100, 10, 3),   # Non-power-of-2
    ])
    def test_build_and_forward_pass(self, affine_architect, input_size, output_size, num_layers):
        """Test various configurations and ensure the output shape is correct."""
        model = affine_architect.build(input_size, output_size, nn.ReLU(), num_layers)
        test_tensor = torch.randn(8, input_size)
        output_tensor = model(test_tensor)
        assert output_tensor.shape == (8, output_size)

    def test_detailed_layer_sizes(self, affine_architect):
        """
        Verify that the in/out features of each layer follow the geometric progression.
        """
        input_size, output_size, num_layers = 512, 32, 4
        
        model = affine_architect.build(
            input_vector_size=input_size,
            output_vector_size=output_size,
            activation=nn.ReLU(),
            num_layers=num_layers
        )
        
        expected_steps = np.geomspace(input_size, output_size, num_layers + 1).round().astype(int)
        
        linear_layers = [m for m in model if isinstance(m, nn.Linear)]
        assert len(linear_layers) == num_layers

        for i, layer in enumerate(linear_layers):
            assert layer.in_features == expected_steps[i]
            assert layer.out_features == expected_steps[i + 1]

    @pytest.mark.parametrize("activation_class", [nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.ELU])
    def test_various_activations(self, affine_architect, activation_class):
        """Test that different activation functions can be used."""
        activation = activation_class()
        model = affine_architect.build(100, 20, activation, 3)
        
        # Expected: Linear, Activation, Linear, Activation, Linear
        assert len(model) == 5
        assert isinstance(model[1], activation_class)
        assert isinstance(model[3], activation_class)
        # Ensure the final layer does not have an activation unless specified
        assert isinstance(model[4], nn.Linear)

# --- Tests for ConvArchitect ---

class TestConvArchitect:
    """Grouped tests for the ConvArchitect class."""
    
    class TestSolveLayerParams:
        """Unit tests for the private _solve_layer_params helper method."""
        
        def test_encoder_simple(self, conv_architect):
            """Test a simple encoder case (e.g., 28 -> 14)."""
            k, s, p = conv_architect._solve_layer_params(28, 14, 'encoder')
            # Expected solution: kernel=2, stride=2, padding=0
            assert (k, s, p) == (2, 2, 0)
            
        def test_decoder_simple(self, conv_architect):
            """Test a simple decoder case (e.g., 7 -> 14)."""
            k, s, p = conv_architect._solve_layer_params(7, 14, 'decoder')
            # Expected solution: kernel=2, stride=2, padding=0
            assert (k, s, p) == (2, 2, 0)

        def test_identity_transform(self, conv_architect):
            """Test an identity spatial transformation (e.g., 32 -> 32)."""
            # Should find a solution with stride=1, e.g., k=3, s=1, p=1
            k, s, p = conv_architect._solve_layer_params(32, 32, 'encoder')
            assert s == 1
            assert ((32 + 2*p - k) / s) + 1 == 32

        def test_failure_case(self, conv_architect):
            """Test that an impossible transformation raises a ValueError."""
            with pytest.raises(ValueError):
                # e.g., huge upsampling with small max kernel/stride
                conv_architect._solve_layer_params(5, 50, 'decoder', max_stride=2, max_kernel=3)

    @pytest.mark.parametrize("input_shape, output_shape, num_layers", [
        ((3, 64, 64), (128, 8, 8), 3),  # Encoder
        ((128, 8, 8), (3, 64, 64), 3),  # Decoder
        ((1, 28, 28), (32, 10, 10), 2), # Encoder, non-power-of-2
        ((16, 7, 7), (1, 28, 28), 2)    # Decoder, non-power-of-2
    ])
    def test_build_and_forward_pass(self, conv_architect, input_shape, output_shape, num_layers):
        """Test various conv configurations and ensure the output shape is correct."""
        model = conv_architect.build(input_shape, output_shape, nn.ReLU(), num_layers)
        
        # Check that the model type is correct (encoder vs decoder)
        mode = 'encoder' if output_shape[1] < input_shape[1] else 'decoder'
        LayerClass = nn.Conv2d if mode == 'encoder' else nn.ConvTranspose2d
        conv_layers = [m for m in model if isinstance(m, nn.modules.conv._ConvNd)]
        assert all(isinstance(layer, LayerClass) for layer in conv_layers)
        
        # Check forward pass
        batch_size = 4
        test_tensor = torch.randn(batch_size, *input_shape)
        output_tensor = model(test_tensor)
        assert output_tensor.shape == (batch_size, *output_shape)

    def test_non_square_transformation(self, conv_architect):
        """Test a transformation between non-square shapes."""
        input_shape = (4, 32, 64)
        output_shape = (64, 8, 16)
        num_layers = 2
        
        model = conv_architect.build(input_shape, output_shape, nn.ELU(), num_layers)
        
        test_tensor = torch.randn(2, *input_shape)
        output_tensor = model(test_tensor)
        assert output_tensor.shape == (2, *output_shape)

    def test_identity_spatial_change_channels(self, conv_architect):
        """Test changing channels while keeping spatial dimensions the same."""
        input_shape = (3, 32, 32)
        output_shape = (64, 32, 32) # Note: out_h == in_h
        
        # The architect will default to 'decoder' mode, but it should work.
        model = conv_architect.build(input_shape, output_shape, nn.ReLU(), 3)
        
        conv_layers = [m for m in model if isinstance(m, nn.modules.conv._ConvNd)]
        # All layers should have stride=1 to maintain spatial dimensions
        for layer in conv_layers:
            assert layer.stride == (1, 1)
        
        test_tensor = torch.randn(4, *input_shape)
        output_tensor = model(test_tensor)
        assert output_tensor.shape == (4, *output_shape)
        
    def test_raises_error_on_impossible_build(self):
        """Test that build() fails if _solve fails, using a restrictive architect."""
        restrictive_architect = ConvArchitect(max_stride=1, max_kernel=1)
        with pytest.raises(ValueError, match="Could not find valid"):
            restrictive_architect.build((1, 10, 10), (1, 5, 5), nn.ReLU(), 1)
    
    def test_known_limitation_mixed_mode_fails(self, conv_architect):
        """
        Test that the architect fails on mixed-mode transformations (e.g., H shrinks, W grows).
        This is an expected failure due to the single 'mode' design.
        """
        input_shape = (3, 32, 16)
        output_shape = (16, 16, 32) # H is encoded, W is decoded
        
        # The architect will choose 'encoder' mode based on H, and then fail to solve for W.
        with pytest.raises(ValueError):
            conv_architect.build(input_shape, output_shape, nn.ReLU(), 2)

    def test_intermediate_channel_sizes(self, conv_architect):
        """Verify that the in/out channels of each layer follow the linear progression."""
        input_shape = (3, 32, 32)
        output_shape = (128, 8, 8)
        num_layers = 4
        
        model = conv_architect.build(input_shape, output_shape, nn.ReLU(), num_layers)
        expected_c_steps = np.linspace(
            input_shape[0], output_shape[0], num_layers + 1
        ).round().astype(int)

        conv_layers = [m for m in model if isinstance(m, nn.Conv2d)]
        assert len(conv_layers) == num_layers

        for i, layer in enumerate(conv_layers):
            assert layer.in_channels == expected_c_steps[i]
            assert layer.out_channels == expected_c_steps[i+1]
