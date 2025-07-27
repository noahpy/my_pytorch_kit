
import torch

class TotalOptimizer:

    """
    Unifies the optimizer, the scheduler, gradient clipping and more.
    """

    def __init__(self, model, optimizer, scheduler=None, max_norm=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_norm = max_norm

    def step(self):
        if self.max_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


def get_optimizer_total_optimizer(model, hparams, use_scheduler=False, use_grad_clip=False) -> TotalOptimizer:
    """
    Get the optimizer and the scheduler

    Parameters
    ----------
    model: torch.nn.Module
        The model to train.
    hparams: dict
        Hyperparameters.
        Should specify hparams["learning_rate"] and hparams["optimizer"].
        If not, the respective default values are 1e-3 and "Adam".
    use_scheduler: bool
        Whether to use a scheduler.
        If True, the scheduler will initialized using hparams["step_size"] and hparams["gamma"].
        Respective default values are 10 and 0.8.
    use_grad_clip: bool
        Whether to use gradient clipping.
        If True, the gradient clipping will initialized using hparams["max_norm"].
        Respective default value is 1.0

    Returns
    -------
    TotalOptimizer
    """

    DEFAULT_STEP_SIZE = 10
    DEFAULT_GAMMA = 0.8
    DEFAULT_MAX_NORM = 1.0
    DEFAULT_OPTIMIZER = "Adam"
    DEFAULT_LEARNING_RATE = 1e-3


    learning_rate = hparams.get("learning_rate", DEFAULT_LEARNING_RATE)
    scheduler = None
    max_norm = None

    if "optimizer" not in hparams:
        hparams["optimizer"] = DEFAULT_OPTIMIZER
    if hparams["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    if use_scheduler:
        step_size = hparams.get("step_size", DEFAULT_STEP_SIZE)
        gamma = hparams.get("gamma", DEFAULT_GAMMA)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if use_grad_clip:
        max_norm = hparams.get("max_norm", DEFAULT_MAX_NORM)

    return TotalOptimizer(model, optimizer, scheduler, max_norm)
