
from my_pytorch_kit.evaluation.evaluation import Evaluator


class ReconstructionEvaluator(Evaluator):

    def __init__(self, criterion):
        """
        Parameters
        ----------
        criterion: torch.nn.Module
            A loss function.
        """
        super().__init__()
        self.acc = 0
        self.batch_count = 0
        self.criterion = criterion

    def evaluate_batch(self, model, batch):
        if isinstance(batch, tuple) or isinstance(batch, list):
            x = batch[0]
        else:
            x = batch
        x_hat = model(x)
        loss = self.criterion(x_hat, x)
        return loss

    def accumulate_result(self, result):
        self.batch_count += 1
        self.acc = (self.acc * (self.batch_count - 1) + result) / self.batch_count

    def get_result(self):
        return self.acc

    def on_eval(self):
        self.acc = 0
        self.batch_count = 0
