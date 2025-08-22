import torch
import torch.nn as nn
from abc import abstractmethod
import os
import inspect
from typing import Optional, Union, Dict

class BaseModel(nn.Module):
    """
    Abstract base class for all models.
    Requires implementation of the calc_loss function.
    Implements weight intialization functions.
    """

    def __init__(self, **args):
        super().__init__()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        init_signature = inspect.signature(cls.__init__)
        parameters = list(init_signature.parameters.values())

        # Check if the __init__ method accepts variable keyword arguments (**kwargs)
        if not any(param.kind == inspect.Parameter.VAR_KEYWORD
                   for param in init_signature.parameters.values()):
            error = f"The __init__ method of '{cls.__name__}' must accept **kwargs."+\
                "\nPlease add **kwargs to the __init__ method."+\
                "\nThis is because the Tuner needs to be able to initialize different models."
            raise TypeError(error)

        # Check all parameters after 'self'
        # The first parameter is expected to be 'self'
        for param in parameters[1:]:
            # Disallow positional-only and positional-or-keyword arguments
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                raise TypeError(
                    f"The __init__ method of '{cls.__name__}' can only have "
                    f"keyword-only arguments, but '{param.name}' is not."
                )

    @abstractmethod
    def calc_loss(self, batch, criterion, **kwargs) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculates the loss for a given batch and criterion.

        Parameters
        ----------
        batch: torch.Tensor
            A batch of data.
        criterion: torch.nn.Module
            A loss function.
        **kwargs: dict
            Additional arguments if needed.

        Returns
        -------
        torch.Tensor
            The loss for the batch.
        OR
        dict[str, torch.Tensor]
            A dictionary of losses.
            Losses should be keyed by name.
            The total loss should be keyed by 'loss'.
        """
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

    def proper_weight_init(self, sample = None):
        """
        Proper default weight initialization.
        If there are uninitialized lazy layers, initialize them by providing a sample.
        If there are more ReLU-like activations than sigmoid-like activations, initialize the linear layers with Kaiming initialization,
        else Xavier initialization.

        Parameters
        ----------
        sample: torch.Tensor
            A sample of the data.
        """

        has_lazy_layers_before = False
        for module in self.modules():
            if isinstance(module, nn.modules.lazy.LazyModuleMixin) and module.has_uninitialized_params():
                has_lazy_layers_before = True
                break

        if has_lazy_layers_before:
            if sample is None:
                raise ValueError("Sample must be provided if there are uninitialized lazy layers.")

            try:
                with torch.no_grad():
                    self(sample)
            except Exception as e:
                raise ValueError(f"Error during initialization: {e}")


        amount_relu_like_activations = 0
        amount_sigmoid_like_activations = 0
        for module in self.modules():
            if isinstance(module, (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.ELU, nn.PReLU, nn.GELU)):
                amount_relu_like_activations += 1
            elif isinstance(module, (nn.Sigmoid, nn.Tanh)):
                amount_sigmoid_like_activations += 1

        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'norm' in name:  # Layer norm weights
                    nn.init.ones_(param)
                elif param.dim() >= 2:  # For matrices (linear layers)
                    if amount_relu_like_activations > amount_sigmoid_like_activations:
                        torch.nn.init.kaiming_uniform_(param, nonlinearity='relu')
                    else:
                        torch.nn.init.xavier_normal_(param)
                else:  # For vectors
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)  # Always initialize biases to zero


    def save_model(self, path, hparams=None):
        """
        Save models state to given path.
        Will also safe hyperparameters if provided.
        Will create directories if necessary.

        Parameters
        ----------
        path: str
            Path to save model to.
        hparams: dict
            Dictionary of hyperparameters.
        """

        base_dir = os.path.dirname(path)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        if hparams is None:
            torch.save(self.state_dict(), path)
        else:
            torch.save({"state_dict": self.state_dict(), "hparams": hparams}, path)

    def load_model(self, path) -> Optional[dict]:
        """
        Load model from given path.
        Will return hyperparameters if the model was saved with them.

        Parameters
        ----------
        path: str
            Path to load model from.
        """
        obj = torch.load(path, weights_only=False)
        hparams = None
        if "hparams" in obj:
            hparams = obj["hparams"]
            self.__init__(**hparams)
            state_dict = obj["state_dict"]
        else:
            state_dict = obj
        self.load_state_dict(state_dict)

        return hparams
