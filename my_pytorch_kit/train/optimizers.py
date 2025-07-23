
import torch

class TotalOptimizer:

    """
    Unifies the optimizer and the scheduler
    """

    def __init__(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def step(self):
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


def get_optimizer_total_optimizer(model, hparams, use_scheduler=False) -> TotalOptimizer:
    """
    Get the optimizer and the scheduler

    Parameters
    ----------
    model: torch.nn.Module
        The model to train.
    hparams: dict
        Hyperparameters.
    use_scheduler: bool
        Whether to use a scheduler.

    Returns
    -------
    TotalOptimizer
    """
    learning_rate = hparams.get("learning_rate", 1e-3)
    scheduler = None
    if "optimizer" not in hparams:
        hparams["optimizer"] = "Adam"
    if hparams["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    if use_scheduler:
        step_size = hparams.get("step_size", 10)
        gamma = hparams.get("gamma", 0.8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    return TotalOptimizer(optimizer, scheduler)
