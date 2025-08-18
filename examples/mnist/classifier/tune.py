
"""
This file will showcase the usage of the Tuner class,
by training a simple model on the MNIST dataset.

!! torchvision is required to be installed !!
"""

import torch.nn as nn

from mnist.classifier.example import MyMnistModel, MnistAccEvaluator
from mnist.utils.mnist_utils import get_mnist_loaders
from my_pytorch_kit.train.tune import Tuner


#++++++++++++++  Here comes the main code ++++++++++++++++#


if __name__ == "__main__":

    """
    The following code will showcase three search algorithms implemented in the Tuner.tune() method.
    One could also use the Tuner.grid_search(), Tuner.random_search() and Tuner.dynamic_search() methods
    instead of the single tune() method, respectively.
    Comment out sections of code if you would like.
    """

    # intialize dataloaders
    batch_size = 64
    train_loader, val_loader, test_loader = get_mnist_loaders(batch_size)

    # intialize Tuner
    tuner = Tuner(MyMnistModel)

    # initialize evaluator
    evaluator = MnistAccEvaluator()

    #++++++++++++ Grid Search +++++++++++++#

    # set search space of hyperparameters
    search_space = {
        "learning_rate": [1e-3, 1e-4],
        "epochs": 1,
        "patience": 5,
        "optimizer_method": ["Adam", "SGD"],
        "loss_func": nn.CrossEntropyLoss(),
        "h1_size": 256,
    }

    # train model
    model, best_config, results = tuner.tune(train_loader, val_loader, search_space)

    result = evaluator.evaluate(model, test_loader)
    print(f"Model accuracy on test set doing grid search: {result*100:.2f}%")

    #++++++++++++ Random Search +++++++++++++#

    # set sample space of hyperparameters
    sample_space = {
        "learning_rate": ([1e-3, 1e-4], 'log'),
        "epochs": 1,
        "patience": 5,
        "optimizer_method": (["Adam", "SGD"], 'item'),
        "loss_func": nn.CrossEntropyLoss(),
        "h1_size": ([128, 256], 'int'),
    }

    # train model
    num_search = 6
    model, best_config, results = tuner.tune(train_loader, val_loader, sample_space, 
                                             num_search=num_search, mode='random')

    result = evaluator.evaluate(model, test_loader)
    print(f"Model accuracy on test set doing random search: {result*100:.2f}%")

    #++++++++++++ Random Dynamic Search +++++++++++++#


    num_search = 10
    ranks_considered = 3
    check_multiplier = 2

    # train model
    model, best_config, results = tuner.tune(train_loader, val_loader, sample_space,
                                             num_search=num_search,
                                             ranks_considered=ranks_considered,
                                             check_multiplicant=check_multiplier,
                                             mode='dynamic')

    result = evaluator.evaluate(model, test_loader)
    print(f"Model accuracy on test set doing random dynamic search: {result*100:.2f}%")


    print("Done!")
