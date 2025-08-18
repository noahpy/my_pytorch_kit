
from torch import nn
import numpy as np
from typing import Tuple, List


class AffineArchitect:
    """
    A class to automatically architect a sequence of affine layers to match a
    desired output shape.
    """

    def __init__(self):
        super().__init__()


    def build(self, 
              input_vector_size: int,
              output_vector_size: int,
              activation: nn.Module,
              num_layers: int,
              last_actication: nn.Module = None) -> nn.Sequential:
        """
        Builds and returns a torch.nn.Sequential model with the specified architecture.

        Args:
            input_vector_size (int): The input vector size.
            output_vector_size (int): The desired output vector size.
            activation (nn.Module): The activation function to use.
            num_layers (int): The number of layers to use.
            last_actication (nn.Module, optional): The last activation function to use.

        Returns:
            nn.Sequential: A model containing the architected layers.

        """

        if input_vector_size == output_vector_size:
            return nn.Identity()

        neuron_steps = np.geomspace(input_vector_size, output_vector_size, num_layers + 1).round().astype(int)

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(neuron_steps[i], neuron_steps[i + 1]))
            if i != num_layers - 1:
                layers.append(activation)

        if last_actication is not None:
            layers.append(last_actication)

        return nn.Sequential(*layers)

class ConvArchitect:
    """
    A class to automatically architect a sequence of 2D convolutional or
    transposed convolutional layers to match a desired output shape.
    """

    def __init__(self, max_stride: int = 5, max_kernel: int = 7):
        super().__init__()
        self.max_stride = max_stride
        self.max_kernel = max_kernel

    def _solve_layer_params(self, 
                            in_dim: int, 
                            out_dim: int, 
                            mode: str, 
                            max_stride: int = None,
                            max_kernel: int = None) -> Tuple[int, int, int]:
        """
        Finds valid (kernel, stride, padding) for a single dimension change.

        It iterates through possible strides and kernels to find a combination
        that results in the target output dimension with a valid integer padding.
        """

        if max_stride is None:
            max_stride = self.max_stride
        if max_kernel is None:
            max_kernel = self.max_kernel


        for stride in range(1, max_stride + 1):
            for kernel_size in range(stride, max_kernel + 1):
                # For Conv2d: out = floor((in + 2*p - k) / s) + 1
                # Rearranging for padding p: 2*p = (out - 1)*s - in + k
                if mode == 'encoder':
                    required_padding_float = ((out_dim - 1) * stride - in_dim + kernel_size) / 2
                # For ConvTranspose2d: out = (in - 1)*s - 2*p + k
                # Rearranging for padding p: 2*p = (in - 1)*s - out + k
                else:  # 'decoder'
                    required_padding_float = ((in_dim - 1) * stride - out_dim + kernel_size) / 2

                # Check if the required padding is a non-negative integer
                if required_padding_float >= 0 and required_padding_float.is_integer():
                    return (kernel_size, stride, int(required_padding_float))

        raise ValueError(
            f"Could not find valid (kernel, stride, padding) for transformation "
            f"{in_dim} -> {out_dim} in '{mode}' mode. "
            f"Try increasing num_layers or adjusting target shape."
        )

    def build(self,
              input_shape: Tuple[int, int, int],
              output_shape: Tuple[int, int, int],
              activation: nn.Module,
              num_layers: int, 
              last_actication: nn.Module = None) -> nn.Sequential:
        """
        Builds and returns a torch.nn.Sequential model with the specified architecture.

        Args:
            input_shape (Tuple[int, int, int]): The input shape (channels, height, width).
            output_shape (Tuple[int, int, int]): The desired output shape (channels, height, width).
            activation (nn.Module): The activation function to use inbetween conv layers.
            num_layers (int): The number of layers to use.
            last_actication (nn.Module): The activation function to use for the last layer.

        Returns:
            nn.Sequential: A model containing the architected layers.
        """
        in_c, in_h, in_w = input_shape
        out_c, out_h, out_w = output_shape

        # detect mixed-mode transforms
        if (in_h < out_h and in_w > out_w) or (in_h > out_h and in_w < out_w):
            raise ValueError("Mixed-mode transforms are not supported. Try using a different architecture.")

        # Determine if we are encoding (shrinking) or decoding (expanding)
        # This assumes a consistent transformation across the network
        mode = 'encoder' if out_h < in_h else 'decoder'
        LayerClass = nn.Conv2d if mode == 'encoder' else nn.ConvTranspose2d

        # Create geometric steps for spatial and linear steps for channel dimensions
        h_steps = np.geomspace(in_h, out_h, num_layers + 1).round().astype(int)
        w_steps = np.geomspace(in_w, out_w, num_layers + 1).round().astype(int)
        c_steps = np.linspace(in_c, out_c, num_layers + 1).round().astype(int)

        layers = []
        for i in range(num_layers):
            # Get dimensions for the current layer
            current_in_c, current_out_c = c_steps[i], c_steps[i+1]
            current_in_h, current_out_h = h_steps[i], h_steps[i+1]
            current_in_w, current_out_w = w_steps[i], w_steps[i+1]

            # Solve for parameters for this layer. Kernel and stride must be the same
            # for H and W, but padding can be different.
            k_h, s_h, p_h = self._solve_layer_params(current_in_h, current_out_h, mode)
            k_w, s_w, p_w = self._solve_layer_params(current_in_w, current_out_w, mode)

            layers.append(
                LayerClass(
                    in_channels=current_in_c,
                    out_channels=current_out_c,
                    kernel_size=(k_h, k_w),
                    stride=(s_h, s_w),
                    padding=(p_h, p_w)
                )
            )

            if i < num_layers - 1:
                layers.append(activation)

        if last_actication is not None:
            layers.append(last_actication)

        return nn.Sequential(*layers)
