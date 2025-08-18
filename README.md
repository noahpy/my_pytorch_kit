![Testing workflow](https://github.com/noahpy/pytorch_toolkit/actions/workflows/ci.yaml/badge.svg)

## My Pytorch Toolkit
Userful pytorch toolkit for training models, replacing boilerplate code.
Provides functions for training, modeling and evaluating models.
Also provides several architecture implementations.

## Installation
### From PyPI

```bash
pip install my-pytorch-kit
```

### From source
Clone this repo and run `pip install .`.  
Then, you can import the module `my_pytorch_kit`.

## Usage

This package revolves around the `BaseModel`, `Trainer` and `Evaluator` classes, which are extended to model, train and evaluate a model respectively.  
```mermaid
graph TD
    subgraph "Usage Workflow"
        A["<b>Define Model</b><br/>(extends BaseModel)"]
        B["<b>Define Data</b><br/>(Dataset / DataLoader)"]
        G["<b>Extras</b><br/>(Optimizer, Tensorboard)"]
        C["<b>Initialize Trainer</b><br/>(extends Trainer)"]
        D["<b>Initialize Evaluator</b><br/>(extends Evaluator)"]
        E["trainer.train()"]
        F["evaluator.evaluate()"]
        G["<b>Intitialize Tuner</b><br/>(Hyperparameter Tuning)"]
        H["tuner.tune()"]
        I["<b>Provided Architectures</b><br/>(Classifier, AE, VAE, ...)"]

    end

    %% Define node relationships
    A --> C
    B --> C
    I --> A
    A --> D
    B --> D
    C --> E
    D --> F
    C --> G
    G --> H

    %% Style the nodes
    style A fill:#fbe,stroke:#333,stroke-width:2px
    style B fill:#fbe,stroke:#333,stroke-width:2px
    style G fill:#ffc,stroke:#333,stroke-width:2px
    style C fill:#cde,stroke:#333,stroke-width:2px
    style D fill:#cde,stroke:#333,stroke-width:2px
    style E fill:#cfc,stroke:#333,stroke-width:2px
    style F fill:#cfc,stroke:#333,stroke-width:2px
    style G fill:#cde,stroke:#333,stroke-width:2px
    style H fill:#cfc,stroke:#333,stroke-width:2px
```

Furthermore, this package provides architecture implementations and modelling utilities.
Currently implemented architectures include:
- Classifiier
- Autoencoder
- Variational Autoencoder

For an **initial** guide, see the `examples/mnist/classifier/example.py` file.


## Development
Clone this repo and run `pip install -e .[dev]`.  
Run pytest in the root directory to run tests.  

