
from my_pytorch_kit.evaluation.evaluation import Evaluator


class ReconstructionEvaluator(Evaluator):

    def __init__(self, criterion, only_accumulate=False):
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
        self.only_accumulate = only_accumulate

    def evaluate_batch(self, model, batch):
        if isinstance(batch, tuple) or isinstance(batch, list):
            x = batch[0]
        else:
            x = batch
        x_hat = model(x)
        loss = self.criterion(x_hat, x)
        return loss

    def accumulate_result(self, result):
        if self.only_accumulate:
            self.acc.append(result.item())
            return
        self.batch_count += 1
        self.acc = (self.acc * (self.batch_count - 1) + result) / self.batch_count

    def get_result(self):
        return self.acc

    def on_eval(self):
        if self.only_accumulate:
            self.acc = []
        else:
            self.acc = 0
        self.batch_count = 0
