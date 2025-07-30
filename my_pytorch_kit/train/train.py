from tqdm import tqdm
import torch
import numpy as np
from my_pytorch_kit.train.optimizers import TotalOptimizer
from my_pytorch_kit.model.models import BaseModel
from my_pytorch_kit.train.tensorboard import get_tensorboard_logger

def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc)



class Trainer:
    """
    A class for training a model.
    """

    def __init__(self, model, train_loader, val_loader, tb_logger = None, initialize_tb = True):
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
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        if tb_logger is None and initialize_tb:
            tb_logger = get_tensorboard_logger()
        self.tb_logger = tb_logger

        self.isTotalOptimizer = True



    def train(self, loss_func, optimizer,
              epochs=10,
              loss_cutoff_rate=0.1,
              patience=None,
              name="model",
              override_instance_errors=False, **kwargs) -> float:
        """
        Train a model and log loss to tensorboard.
        If interrupted by KeyboardInterrupt, exit gracefully.
        Also compatible with torch.nn.Model as a model and torch.optim.Optimizer as an optimizer,
        if override_instance_errors is True.
        Returns best validation loss.

        Parameters
        ----------
        loss_func: torch.nn.Module
            The loss function.
        optimizer: torch.optim.Optimizer
            The optimizer.
        epochs: int
            The number of epochs to train for. (default: 10)
        loss_cutoff_rate: float
            The rate at which to cut off the loss. (default: 0.1)
        patience: int
            The number of epochs to wait before early stopping. (default: None)
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
                    loss = self.model.calc_loss(batch, loss_func)
                    loss.backward()
                    optimizer.step()


                    training_loss.append(loss.item())
                    last_cutoff_loss = np.mean(training_loss[-loss_cutoff:])

                    # Update the progress bar.
                    self.set_training_post_fix(training_loop, last_cutoff_loss, optimizer)

                    # Update the tensorboard logger.
                    self.add_tb_scalar(f'{name}/train_loss', loss.item(), epoch * len(self.train_loader) + train_iteration)

                # VALIDATION
                self.model.eval()
                val_loop = create_tqdm_bar(self.val_loader, desc=f'Validation Epoch [{epoch + 1}/{epochs}]')

                with torch.no_grad():
                    for val_iteration, batch in val_loop:
                        loss = self.model.calc_loss(batch, loss_func)
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
        loop.set_postfix(curr_train_loss = "{:.5f}".format(loss),
                         lr = "{:.5f}".format(lr))

    def set_validation_post_fix(self, loop, loss):
        loop.set_postfix(val_loss = "{:.5f}".format(loss))
