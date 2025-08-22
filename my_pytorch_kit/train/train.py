from tqdm import tqdm
import torch
import numpy as np
from my_pytorch_kit.train.optimizers import TotalOptimizer
from my_pytorch_kit.model.models import BaseModel
from my_pytorch_kit.train.tensorboard import get_tensorboard_logger
from typing import Callable, Optional

def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable), total=len(iterable), desc=desc, dynamic_ncols=True)



class Trainer:
    """
    A class for training a model.
    """

    def __init__(self, 
                 model: torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 tb_logger: Optional[torch.utils.tensorboard.SummaryWriter] = None,
                 initialize_tb: bool = True,
                 sample_input_shape: Optional[tuple] = None,
                 **kwargs
                 ):
        """
        Initialize the trainer.

        Parameters
        ----------
        model: torch.nn.Module
            The model to train.
        train_loader: torch.utils.data.DataLoader
            The training data loader.
        val_loader: torch.utils.data.DataLoader
            The validation data loader.
        tb_logger: SummaryWriter
            The tensorboard logger.
        initialize_tb: bool
            Whether to initialize the tensorboard logger, if tb_logger is None.
        sample_input_shape: tuple
            The shape of the input to the model.
            If given, will add a graph to the tensorboard logger.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        if tb_logger is None and initialize_tb:
            tb_logger = get_tensorboard_logger()
        self.tb_logger = tb_logger

        self.isTotalOptimizer = True

        if sample_input_shape is not None:
            self.tb_logger.add_graph(self.model, torch.randn(*sample_input_shape))
        else:
            print("Sample input shape not given. Not adding graph to tensorboard logger.")


    def train(self, 
              optimizer: TotalOptimizer,
              loss_func: Callable,
              epochs: int = 10,
              loss_cutoff_rate: float = 0.1,
              patience: Optional[int] = None,
              epoch_function: Optional[Callable] = None,
              name: str = "model",
              override_instance_errors: bool = False,
              **kwargs) -> float:
        """
        Train a model and log loss to tensorboard.
        If interrupted by KeyboardInterrupt, exit gracefully.
        Also compatible with torch.nn.Model as a model and torch.optim.Optimizer as an optimizer,
        if override_instance_errors is True.
        Returns best validation loss.

        Parameters
        ----------
        optimizer: torch.optim.Optimizer
            The optimizer.
        loss_func: torch.nn.Module
            The loss function.
        epochs: int
            The number of epochs to train for. (default: 10)
        loss_cutoff_rate: float
            The rate at which to cut off the loss. (default: 0.1)
        patience: int
            The number of epochs to wait before early stopping. (default: None)
        epoch_function: Callable
            Function called after each epoch. (default: None)
            Following named parameters are passed to it:
                - epoch: int
                - epochs: int
                - model: model
                - lr: learning rate
                - tl: training loss
                - vl: validation loss
                - kwargs: kwargs passed to this initializer
        name: str
            The name of the model.
        override_instance_errors: bool
            Whether to ignore instance errors.

        Returns
        -------
        best_val_loss: float
            The best validation loss.
        """

        if not issubclass(type(self.model), BaseModel) and not override_instance_errors:
            raise ValueError(f"model with type {type(self.model)} must be a subclass of BaseModel")

        if not isinstance(optimizer, TotalOptimizer):
            if not override_instance_errors:
                raise ValueError("Optimizer must be an instance of TotalOptimizer")
            self.isTotalOptimizer = False

        loss_cutoff = int(len(self.train_loader) * loss_cutoff_rate)
        patience_counter = 0
        best_val_loss = float('inf')

        try:
            for epoch in range(epochs):
                training_loss = []
                validation_loss = []

                # TRAINING
                self.model.train()
                training_loop = create_tqdm_bar(self.train_loader, desc=f'Training Epoch [{epoch + 1}/{epochs}]')
                for train_iteration, batch in training_loop:
                    optimizer.zero_grad()
                    returned_loss = self.model.calc_loss(batch, loss_func)
                    if type(returned_loss) is dict:
                        loss = returned_loss['loss']
                    else:
                        loss = returned_loss
                    loss.backward()
                    optimizer.step()


                    training_loss.append(loss.item())
                    last_cutoff_loss = np.mean(training_loss[-loss_cutoff:])

                    # Update the progress bar.
                    self.set_training_post_fix(training_loop, last_cutoff_loss, optimizer)

                    # Update the tensorboard logger.
                    self.add_tb_scalar(f'{name}/train_loss', loss.item(), epoch * len(self.train_loader) + train_iteration)

                    if type(returned_loss) is dict:
                        for loss_name, loss_value in returned_loss.items():
                            if loss_name != 'loss':
                                self.add_tb_scalar(f'{name}/loss_components/{loss_name}', loss_value.item() / loss.item(), epoch * len(self.train_loader) + train_iteration)



                # VALIDATION
                self.model.eval()
                val_loop = create_tqdm_bar(self.val_loader, desc=f'Validation Epoch [{epoch + 1}/{epochs}]')

                with torch.no_grad():
                    for val_iteration, batch in val_loop:
                        returned_loss = self.model.calc_loss(batch, loss_func)
                        if type(returned_loss) is dict:
                            loss = returned_loss['loss']
                        else:
                            loss = returned_loss
                        validation_loss.append(loss.item())
                        last_cutoff_loss = np.mean(validation_loss[-loss_cutoff:])

                        # Update the progress bar.
                        self.set_validation_post_fix(val_loop, last_cutoff_loss)

                        # Update the tensorboard logger.
                        self.add_tb_scalar(f'{name}/val_loss', last_cutoff_loss, epoch * len(self.val_loader) + val_iteration)


                # best val loss check
                total_val_loss = np.mean(validation_loss)
                if total_val_loss < best_val_loss:
                    best_val_loss = total_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                # patience check
                if patience is not None and patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}.")
                    break

                if epoch_function:
                    epoch_function(
                        epoch = epoch,
                        epochs = epochs,
                        model = self.model,
                        lr = optimizer.optimizer.param_groups[0]['lr'],
                        tl = np.mean(training_loss),
                        vl = np.mean(validation_loss),
                        **kwargs
                    )


        except KeyboardInterrupt:
            print("Training interrupted by user.")

        return best_val_loss



    def add_tb_scalar(self, *args, **kwargs):
        if self.tb_logger is not None:
            self.tb_logger.add_scalar(*args, **kwargs)

    def set_training_post_fix(self, loop, loss, optimizer):
        if self.isTotalOptimizer:
            lr = optimizer.optimizer.param_groups[0]['lr']
        else:
            lr = optimizer.param_groups[0]['lr']
        loop.set_postfix(loss = "{:.4f}".format(loss),
                         lr = "{:.4f}".format(lr))

    def set_validation_post_fix(self, loop, loss):
        loop.set_postfix(loss = "{:.4f}".format(loss))
