
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


def get_optimizer_total_optimizer(model, learning_rate=1e-3,
                                  optimizer_method="Adam",
                                  step_size=10,
                                  gamma=0.8,
                                  max_norm=1.0,
                                  use_scheduler=False,
                                  use_grad_clip=False,
                                  optimizer_kwargs={},
                                  **kwargs) -> TotalOptimizer:
    """
    Get the optimizer and the scheduler

    Parameters
    ----------
    model: torch.nn.Module
        The model to train.
    learning_rate: float
        The learning rate.
    optimizer_method: str
        The optimizer method.
        Respective default value is "Adam"
    step_size: int
        The step size for the scheduler.
        Respective default value is 10
    gamma: float
        The gamma for the scheduler.
        Respective default value is 0.8
    max_norm: float
        The max norm for the gradient clipping.
        Respective default value is 1.0
    use_scheduler: bool
        Whether to use a scheduler.
        If True, the scheduler will initialized using step_size and gamma.
        Respective default values are 10 and 0.8.
    use_grad_clip: bool
        Whether to use gradient clipping.
        If True, the gradient clipping will initialized using max_norm.
        Respective default value is 1.0
    optimizer_kwargs: dict
        The kwargs for the optimizer.
        Respective default value is {}

    Returns
    -------
    TotalOptimizer
    """

    scheduler = None

    if optimizer_method == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), learning_rate, **optimizer_kwargs)
    else:
        optimizer = torch.optim.Adam(model.parameters(), learning_rate, **optimizer_kwargs)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if not use_grad_clip:
        max_norm = None

    return TotalOptimizer(model, optimizer, scheduler, max_norm)
