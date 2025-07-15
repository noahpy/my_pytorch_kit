from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from my_pytorch_kit.train.optimizers import TotalOptimizer

from my_pytorch_kit.model.models import BaseModel

def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc)


def train_model(model, hparams, train_loader, val_loader, loss_func, optimizer, tb_logger, name="default"):
    """
    Train a model and log to tensorboard.

    Parameters
    ----------
    model: torch.nn.Module
        The model to train.
    hparams: dict
        Hyperparameters.
    train_loader: torch.utils.data.DataLoader
        The training data loader.
    val_loader: torch.utils.data.DataLoader
        The validation data loader.
    loss_func: torch.nn.Module
        The loss function.
    optimizer: torch.optim.Optimizer
        The optimizer.
    tb_logger: SummaryWriter
        The tensorboard logger.
    """

    if not issubclass(type(model), BaseModel):
        raise ValueError("Model must be a subclass of BaseModel")

    if not isinstance(optimizer, TotalOptimizer):
        raise ValueError("Optimizer must be an instance of TotalOptimizer")

    epochs = hparams.get("epochs", 10)

    loss_cutoff = len(train_loader) // 10

    try:
        for epoch in range(epochs):

            model.train()

            training_loss = []
            validation_loss = []

            # TRAINING
            training_loop = create_tqdm_bar(train_loader, desc=f'Training Epoch [{epoch + 1}/{epochs}]')
            for train_iteration, batch in training_loop:
                optimizer.zero_grad()
                loss = model.calc_loss(batch, loss_func)
                loss.backward()
                optimizer.step()


                training_loss.append(loss.item())
                training_loss = training_loss[-loss_cutoff:]

                # Update the progress bar.
                training_loop.set_postfix(curr_train_loss = "{:.5f}".format(np.mean(training_loss)),
                                          lr = "{:.5f}".format(optimizer.param_groups[0]['lr'])
                                          )

                # Update the tensorboard logger.
                tb_logger.add_scalar(f'classifier_{name}/train_loss', loss.item(), epoch * len(train_loader) + train_iteration)

            # VALIDATION
            model.eval()
            val_loop = create_tqdm_bar(val_loader, desc=f'Validation Epoch [{epoch + 1}/{epochs}]')

            with torch.no_grad():
                for val_iteration, batch in val_loop:
                    loss = model.calc_loss(batch, loss_func)
                    validation_loss.append(loss.item())
                    # Update the progress bar.
                    val_loop.set_postfix(val_loss = "{:.5f}".format(np.mean(validation_loss)))

                    # Update the tensorboard logger.
                    tb_logger.add_scalar(f'classifier_{name}/val_loss', np.mean(validation_loss), epoch * len(val_loader) + val_iteration)
    except KeyboardInterrupt:
        return
