
from my_pytorch_kit.train.train import Trainer
import random
from math import log10
from typing import List, Dict, Tuple
import torch
from itertools import product

class Tuner:

    ALLOWED_RANDOM_SEARCH_PARAMS = ['log', 'int', 'float', 'item', 'logint']

    def __init__(self, model_class, trainer_class = Trainer):
        self.model_class = model_class
        self.trainer_class = trainer_class
    
    def grid_search(self, train_loader, val_loader, grid_search_spaces: Dict[str, List]):
        """
        A simple grid search.
        Searches all combinations of hyperparameters in grid_search_spaces.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            The training data loader.
        val_loader: torch.utils.data.DataLoader
            The validation data loader.
        grid_search_spaces: Dict[str, List]
            Hyperparameter search spaces for grid search.
            Specifies the possible values for each hyperparameter, e.g. 
            {'lr': [1e-3, 1e-4, 1e-5], 'batch_size': [32, 64, 128]}

        Returns
        -------
        best_model: torch.nn.Module
            The model performing best on validation set
        best_config: dict
            The hyperparameter config that performed best
        results: list
            List of tuples (config, val_loss)
        """
        configs = []

        # More general implementation using itertools
        for instance in product(*grid_search_spaces.values()):
            configs.append(dict(zip(grid_search_spaces.keys(), instance)))

        return self.find_best_config(configs, train_loader, val_loader)


    def find_best_config(self, configs: List[Dict], train_loader: torch.utils.data.DataLoader,
                         val_loader: torch.utils.data.DataLoader) -> Tuple[torch.nn.Module, Dict, List]:
        """
        Get a list of hyperparameter configs for random search or grid search,
        trains a model on all configs and returns the one performing best
        on validation set.

        Parameters
        ----------
        configs: list
            List of dict (hyperparameter configs)
        train_loader: torch.utils.data.DataLoader
            The training data loader.
        val_loader: torch.utils.data.DataLoader
            The validation data loader.

        Returns
        -------
        best_model: torch.nn.Module
            The model performing best on validation set
        best_config: dict
            The hyperparameter config that performed best
        results: list
            List of tuples (config, val_loss)
        """

        best_val = float("inf")
        best_config = None
        best_model = None
        results = []

        for i in range(len(configs)):
            print("\nEvaluating Config #{} [of {}]:\n".format(
                (i+1), len(configs)), configs[i])

            model = self.model_class(**configs[i])

            trainer = self.trainer_class(model, train_loader, val_loader)
            val_loss = trainer.train(configs[i])

            results.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_config = configs[i]
                best_model = model


        print("\nSearch done. Best Val Loss = {}".format(best_val))
        print("Best Config:", best_config)
        return best_model, best_config, list(zip(configs, results))


    def random_search_spaces_to_config(self, random_search_spaces: Dict[str, Tuple(List, str)]) -> Dict:
        """"
        Takes search spaces for random search as input; samples accordingly
        from these spaces and returns the sampled hyper-params as a config-object.

        Parameters
        ----------
        random_search_spaces: dict
            Hyperparameter search spaces for random search

        Returns
        -------
        config: dict
            Sampled hyperparameter config
        """

        config = {}

        for key, (rng, mode) in random_search_spaces.items():
            if mode not in self.ALLOWED_RANDOM_SEARCH_PARAMS:
                print("'{}' is not a valid random sampling mode. "
                      "Ignoring hyper-param '{}'".format(mode, key))
            elif mode == "log":
                if rng[0] <= 0 or rng[-1] <= 0:
                    print("Invalid value encountered for logarithmic sampling "
                          "of '{}'. Ignoring this hyper param.".format(key))
                    continue
                sample = random.uniform(log10(rng[0]), log10(rng[-1]))
                config[key] = 10**(sample)
            elif mode == "int":
                config[key] = random.randint(rng[0], rng[-1])
            elif mode == "float":
                config[key] = random.uniform(rng[0], rng[-1])
            elif mode == "item":
                config[key] = random.choice(rng)
            elif mode == "logint":
                if rng[0] <= 0 or rng[-1] <= 0:
                    print("Invalid value encountered for logarithmic sampling "
                          "of '{}'. Ignoring this hyper param.".format(key))
                    continue
                sample = random.uniform(log10(rng[0]), log10(rng[-1]))
                config[key] = int(10**(sample))

        return config
