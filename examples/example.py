
"""
This file will showcase the usage of this package, 
by training a simple model on the MNIST dataset.

!! torchvision is required to be installed !!
"""

import torch.nn as nn
import torch
from torchvision.datasets import MNIST
import torchvision

from my_pytorch_kit.model.models import BaseModel
from my_pytorch_kit.train.train import Trainer
from my_pytorch_kit.evaluation.evaluation import Evaluator
from my_pytorch_kit.train.optimizers import get_optimizer_total_optimizer

class MyMnistModel(BaseModel):
    """
    A simple example model, extending BaseModel class.
    All subclasses of BaseModel must implement calc_loss(), which is 
    then used by the Trainer.train() training loop.
    """

    def __init__(self, *, h1_size=256, **kwargs):
        """
        Watch out!
        In order to be able to be used by the Tuner, the __init__() method
        must accept **kwargs, and use only keyword arguments.
        Thus, the signature must look like this:
            def __init__(self, *, key1=..., key2=..., **kwargs):
        """
        super(MyMnistModel, self).__init__()
        self.model = nn.Sequential(
            nn.LazyLinear(h1_size),
            nn.PReLU(),
            nn.Linear(h1_size, 64),
            nn.PReLU(),
            nn.Linear(64, 10),
        )
        self.use_softmax = False

        # initialize weights
        # as lazy layers are in use, one needs to pass in a random sample
        random_sample = torch.randn((1, 784))
        self.proper_weight_init(random_sample)

    def forward(self, x):
        x = self.model(x)
        if self.use_softmax:
            x = nn.functional.softmax(x, dim=1)
        return x

    def calc_loss(self, batch, criterion):
        x, y = batch
        x = x.view(x.shape[0], 784)
        # convert to one-hot encoding
        y = torch.nn.functional.one_hot(y, num_classes=10).float()
        y_pred = self.forward(x)
        loss = criterion(y_pred, y)
        return loss


class MnistAccEvaluator(Evaluator):
    """
    A simple evaluator, extending Evaluator class.
    The evaluate() function will be called to evaluate the model.
    See Evaluator class for more details.
    """

    def __init__(self):
        super().__init__()
        self.batch_count = 0
        self.result = 0

    def evaluate_batch(self, model, batch):
        model.use_softmax = True
        x, y = batch
        x = x.view(x.shape[0], 784)
        y_pred = model.forward(x)
        correct_prediction = (torch.argmax(y_pred, dim=1) == y).float()
        accuracy = correct_prediction.mean()
        model.use_softmax = False
        return accuracy

    def accumulate_result(self, result):
        # running average over all batches
        self.batch_count += 1
        self.result = (self.result * (self.batch_count - 1) + result) / self.batch_count

    def get_result(self):
        return self.result

    def on_eval(self):
        self.batch_count = 0


def get_mnist_loaders(batch_size=64):
    """
    Download MNIST dataset and split into train, val, test.
    """
    mnist_train_dataset = MNIST(
        root="data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    mnist_test_dataset = MNIST(
        root="data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    train_subset, val_subset = torch.utils.data.random_split(
        mnist_train_dataset, [50000, 10000]
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        mnist_test_dataset, batch_size=3, shuffle=True
    )

    return train_loader, val_loader, test_loader


#++++++++++++++  Here comes the main code ++++++++++++++++#


if __name__ == "__main__":

    # Set hyperparameters
    # The idea is that everything you need to configure, should be configured in this single dict
    hparams = {
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 2,
        "optimizer_method": "Adam",
        "optimizer_kwargs": {"weight_decay": 1e-4},
        "loss_func": nn.CrossEntropyLoss(),
    }

    # create model
    model = MyMnistModel()

    print(model)

    # intialize dataloaders
    train_loader, val_loader, test_loader = get_mnist_loaders(hparams["batch_size"])

    # intialize TotalOptimizer (includes optional scheduler and gradient clipping)
    optimizer = get_optimizer_total_optimizer(model, use_scheduler=False, use_grad_clip=False, **hparams)

    # initialize trainer
    trainer = Trainer(model, train_loader, val_loader)

    # train model
    trainer.train(optimizer, **hparams)

    # initialize evaluator
    evaluator = MnistAccEvaluator()

    # evaluate model
    result = evaluator.evaluate(model, test_loader)

    # print result
    print(f"Model accuracy on test set: {result*100:.2f}%")

    # save model
    model.save_model("models/model.pt")

    # load model
    model.load_model("models/model.pt")

    print("Done!")
