
from torch.utils.tensorboard import SummaryWriter
import os

def get_tensorboard_logger(path="logs"):
    num_of_runs = len(os.listdir(path)) if os.path.exists(path) else 0
    path = os.path.join(path, f'run_{num_of_runs + 1}')
    return SummaryWriter(log_dir=path)
