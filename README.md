![Testing workflow](https://github.com/noahpy/pytorch_toolkit/actions/workflows/ci.yaml/badge.svg)

## Pytorch toolkit
Userful pytorch toolkit for training models.
Provides functions for training, data_processing, modeling and evaluating models.

## Installation
### From PyPI

```bash
pip install my-pytorch-kit
```

### From source
Clone this repo and run `pip install .`.  
Then, you can import the module `my_pytorch_kit`.

## Usage

This package revolves around the `BaseModel`, `Trainer` and `Evaluator` classes, which are used to model, train and evaluate a model respectively by extending them.  
For a detailed guide, see the `examples.py` file :)


## Development
Clone this repo and run `pip install -e .[dev]`.  
Run pytest in the root directory to run tests.  

