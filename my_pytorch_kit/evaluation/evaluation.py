
import torch.nn as nn
from abc import abstractmethod
from typing import Any

class Evaluator(nn.Module):
    """
    Base class for model evaluation.
    """

    def __init__(self):
        super().__init__()


    @abstractmethod
    def evaluate_batch(self, model, batch) -> Any:
        """
        Evaluate a batch and return result metric.

        Parameters
        ----------
        model: torch.nn.Module
            The model to evaluate.
        batch: torch.Tensor
            A batch of data.

        Returns
        -------
        Any 
            The result metric.
        """
        pass

    @abstractmethod
    def accumulate_result(self, result) -> None:
        """
        Accumulate result metric.
        Called after each batch.

        Parameters
        ----------
        result: Any
            The result metric.
        """
        pass

    @abstractmethod
    def get_result(self) -> Any:
        """
        Get the final result metric.

        Returns
        -------
        Any
            The final result metric.
        """
        pass


    def on_eval(self, model):
        """
        Gets called before evaluation begins.

        Parameters
        ----------
        model: torch.nn.Module
            The model to evaluate.
        """
        pass


    def evaluate(self, model, data_loader) -> Any:
        """
        Evaluate the model on the data loader.

        Parameters
        ----------
        model: torch.nn.Module
            The model to evaluate.
        data_loader: torch.utils.data.DataLoader
            The data loader.

        Returns
        -------
        Any
            The result metric.
        """
        self.on_eval(model)
        model.eval()
        for batch in data_loader:
            result = self.evaluate_batch(model, batch)
            self.accumulate_result(result)
        return self.get_result()


