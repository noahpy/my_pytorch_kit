
import random
from math import log10
from typing import List, Dict, Tuple
import torch
from itertools import product

from my_pytorch_kit.train.train import Trainer
from my_pytorch_kit.train.optimizers import get_optimizer_total_optimizer


class Tuner:
    """
    Class for hyperparameter tuning.
    Call tune() for hyper parameter search.

    For an example application, see examples/mnist/classifier/tune.py!
    """

    ALLOWED_RANDOM_SEARCH_PARAMS = ['log', 'int', 'float', 'item', 'logint']

    def __init__(self, model_class, trainer_class = Trainer):
        self.model_class = model_class
        self.trainer_class = trainer_class


    def tune(self, train_loader, val_loader, parameter_space, mode = 'grid', **kwargs):
        """
        Calls the appropriate search method and returns the best model.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            The training data loader.
        val_loader: torch.utils.data.DataLoader
            The validation data loader.
        parameter_space: dict
            Hyperparameter search spaces for grid search or random search
            Should be Dict[str, List] or Dict[str, Tuple[List, str]].
        mode: str
            'grid', 'random' or 'dynamic'
        """

        if mode == 'grid':
            return self.grid_search(train_loader, val_loader, parameter_space)
        elif mode == 'random':
            return self.random_search(train_loader, val_loader, parameter_space, **kwargs)
        elif mode == 'dynamic':
            return self.random_dynamic_search(train_loader, val_loader, parameter_space, **kwargs)
        else:
            raise ValueError("Invalid mode: {}".format(mode))


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

        for key, values in grid_search_spaces.items():
            if not isinstance(values, list):
                grid_search_spaces[key] = [values]

        # More general implementation using itertools
        for instance in product(*grid_search_spaces.values()):
            configs.append(dict(zip(grid_search_spaces.keys(), instance)))

        return self.find_best_config(configs, train_loader, val_loader)


    def random_search(self, train_loader, val_loader, 
                      random_search_spaces: Dict[str, Tuple[List, str]],
                      num_search: int = 10, **kwargs):
        """
        Samples num_search hyper parameter sets within the provided search spaces
        and returns the best model.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            The training data loader.
        val_loader: torch.utils.data.DataLoader
            The validation data loader.
        random_search_spaces: dict
            Hyperparameter search spaces for random search
        num_search: int
            Number of hyperparameter configs to sample (default: 10)

        Returns
        -------
        best_model: torch.nn.Module
            The model performing best on validation set
        best_config: dict
            The hyperparameter config that performed best
        results: list
            List of tuples (config, val_loss)
        """

        # some preprocessing
        for key, value in random_search_spaces.items():
            if not isinstance(value, tuple):
                random_search_spaces[key] = ([value], 'item')

        configs = []
        for _ in range(num_search):
            configs.append(self.random_search_spaces_to_config(random_search_spaces))

        return self.find_best_config(configs, train_loader, val_loader)

    def random_dynamic_search(self, train_loader, val_loader, 
                              random_search_spaces: Dict[str, Tuple[List, str]],
                              num_search=10, ranks_considered=15,
                              check_multiplicant=2, **kwargs):
        """
        Samples num_search hyper parameter sets within the provided search space,
        reducing the search space dynamically by looking at the <ranks_considered> best results,
        if there are at least <check_mutiplicant> * <ranks_considered> configs,
        and returns the best model.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            The training data loader.
        val_loader: torch.utils.data.DataLoader
            The validation data loader.
        num_search: int
            Number of hyperparameter configs to sample
        random_search_spaces: dict
            Hyperparameter search spaces for random search
        ranks_considered: int
            Number of best configs to consider
        check_multiplicant: int
            Multiplicant of ranks_considered

        Returns
        -------
        best_model: torch.nn.Module
            The model performing best on validation set
        best_config: dict
            The hyperparameter config that performed best
        results: list
            List of tuples (config, val_loss)

        """
        config_count = 0
        configs = []

        # stores tuples of (val_loss, config, model)
        ranking = []

        configs.append(self.random_search_spaces_to_config(random_search_spaces))

        results = []

        def print_ranking(ranking):
            for i in range(len(ranking)):
                print("Rank {} with best loss {:.4f}: {}".format(i + 1, ranking[i][0], ranking[i][1]))

        def adapt_search_space(ranking, random_search_space):
            for name, (rng, mode) in random_search_space.items():
                if mode != "item":
                    new_min = min(ranking, key=lambda x: x[1][name])[1][name]
                    new_max = max(ranking, key=lambda x: x[1][name])[1][name]
                    random_search_space[name] = ([new_min, new_max], mode)
            print("New search space:", random_search_space)
            return random_search_space


        while config_count < num_search:
            try:
                config = configs[-1]
                print("\nEvaluating Config #{} [of {}]:\n".format(
                    (config_count), num_search), config)

                model = self.model_class(**config)

                trainer = self.trainer_class(model, train_loader, val_loader)

                optimizer = get_optimizer_total_optimizer(model, **config)

                val_loss = trainer.train(optimizer, **config)

                # add into ranking
                if len(ranking) == 0:
                    ranking.append((val_loss, config, model))
                else:
                    ranking.append((val_loss, config, model))
                    ranking.sort(key=lambda x: x[0])
                    ranking = ranking[:ranks_considered]

                print_ranking(ranking)

                # start adapting search space
                if config_count >= int(check_multiplicant * ranks_considered):
                    random_search_spaces = adapt_search_space(ranking, random_search_spaces)

                config_count += 1
                configs.append(self.random_search_spaces_to_config(random_search_spaces))
            except KeyboardInterrupt:
                break

        best_val, best_model, \
            best_config = ranking[0][0], ranking[0][2], ranking[0][1]

        print("\nSearch done. Best Val Loss = {}".format(best_val))
        print("Best Config:", best_config)
        return best_model, best_config, list(zip(configs, results))


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

            optimizer = get_optimizer_total_optimizer(model, **configs[i])

            val_loss = trainer.train(optimizer, **configs[i])

            results.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_config = configs[i]
                best_model = model


        print("\nSearch done. Best Val Loss = {}".format(best_val))
        print("Best Config:", best_config)
        return best_model, best_config, list(zip(configs, results))


    def random_search_spaces_to_config(self, random_search_spaces: Dict[str, Tuple[List, str]]) -> Dict:
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
                      + "Allowed modes: {}".format(self.ALLOWED_RANDOM_SEARCH_PARAMS)
                      + "Ignoring hyper-param '{}'".format(key))
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
