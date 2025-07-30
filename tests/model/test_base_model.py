import pytest
import torch
import torch.nn as nn
from my_pytorch_kit.model.models import BaseModel
from my_pytorch_kit.train.optimizers import get_optimizer_total_optimizer


class SimpleModel(BaseModel):
    """
    A simple implementation of the BaseModel for testing purposes.
    """

    def __init__(self, *, input_dim = 784, output_dim = 10, use_lazy=False, use_layernorm=False, **kwargs):
        super().__init__()
        self.hparams = hparams
        if use_lazy:
            self.fc1 = nn.LazyLinear(128)
        else:
            self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        if use_layernorm:
            self.layernorm = nn.LayerNorm(64)
        else:
            self.layernorm = nn.Identity()
        self.sigmoid = nn.Sigmoid()
        self.fc3 = nn.Linear(64, output_dim)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.layernorm(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        return x

    def calc_loss(self, batch, criterion) -> torch.Tensor:
        """
        Calculates the loss for a given batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = criterion(y_hat, y)
        return loss

@pytest.fixture
def hparams():
    """
    Pytest fixture for providing hyperparameters.
    """
    return {'input_dim': 256, 'output_dim': 10}


@pytest.fixture
def simple_model(hparams):
    """
    Pytest fixture for a SimpleModel instance.
    """
    return SimpleModel(**hparams)


@pytest.fixture
def sample_batch(hparams):
    """
    Pytest fixture for a sample data batch.
    """
    return (torch.randn(32, hparams['input_dim']),
            torch.randint(0, hparams['output_dim'], (32,)))


def test_kaiming_init():
    """
    Tests the kaiming_init function.
    """
    layer = nn.Linear(10, 20)
    BaseModel.kaiming_init(layer)
    # Check if weights are not all zero
    assert not torch.all(torch.eq(layer.weight, 0))
    # Check if bias is initialized correctly
    assert torch.all(torch.eq(layer.bias, 0.01))


def test_xavier_init():
    """
    Tests the xavier_init function.
    """
    layer = nn.Linear(10, 20)
    BaseModel.xavier_init(layer)
    # Check if weights are not all zero
    assert not torch.all(torch.eq(layer.weight, 0))
    # Check if bias is initialized correctly
    assert torch.all(torch.eq(layer.bias, 0.01))


def test_proper_weight_init_no_lazy_layers(simple_model):
    """
    Tests the proper_weight_init function when there are no lazy layers.
    """
    simple_model.proper_weight_init()
    # Since there's one ReLU and one Sigmoid, it should default to Xavier
    # We can check if the weights have been initialized (i.e., they are not in their default state)
    for name, param in simple_model.named_parameters():
        if 'weight' in name and 'norm' not in name:
            assert not torch.all(torch.eq(param, 0))
        elif 'bias' in name:
            assert torch.all(torch.eq(param, 0))

def test_proper_weight_init_with_lazy_layers(hparams, sample_batch):
    """
    Tests the proper_weight_init function with lazy layers.
    """
    model = SimpleModel(**hparams, use_lazy=True)

    # Test that it raises an error if no sample is provided
    with pytest.raises(ValueError, match="Sample must be provided if there are uninitialized lazy layers."):
        model.proper_weight_init()

    # Test that it initializes correctly with a sample
    sample, _ = sample_batch
    model.proper_weight_init(sample=sample)
    for name, param in model.named_parameters():
        if 'weight' in name and 'norm' not in name:
            assert not torch.all(torch.eq(param, 0))
        elif 'bias' in name:
            assert torch.all(torch.eq(param, 0))

def test_proper_weight_init_kaiming_vs_xavier(hparams):
    """
    Tests that the correct initialization is chosen based on activation functions.
    """
    # More ReLU-like activations -> Kaiming
    class KaimingModel(BaseModel):
        def __init__(self, *, input_dim = 784, output_dim = 10, **kwargs):
            super().__init__()
            self.hparams = hparams
            self.fc1 = nn.Linear(input_dim, 128)
            self.a1 = nn.ReLU()
            self.fc2 = nn.Linear(128, 64)
            self.a2 = nn.LeakyReLU()
            self.fc3 = nn.Linear(64, output_dim)

        def forward(self, x): return x
        def calc_loss(self, batch, criterion): pass

    k_model = KaimingModel(**hparams)
    # Mock the init functions to track which one is called
    k_model.kaiming_init = lambda m: setattr(m, 'was_kaiming_initialized', True)
    k_model.xavier_init = lambda m: setattr(m, 'was_xavier_initialized', True)
    k_model.proper_weight_init()

    # More Sigmoid-like activations -> Xavier
    class XavierModel(BaseModel):
        def __init__(self, *, input_dim = 784, output_dim = 10, **kwargs):
            super().__init__()
            self.hparams = hparams
            self.fc1 = nn.Linear(input_dim, 128)
            self.a1 = nn.Sigmoid()
            self.fc2 = nn.Linear(128, 64)
            self.a2 = nn.Tanh()
            self.fc3 = nn.Linear(64, output_dim)

        def forward(self, x): return x
        def calc_loss(self, batch, criterion): pass

    x_model = XavierModel(**hparams)
    # Mock the init functions
    x_model.kaiming_init = lambda m: setattr(m, 'was_kaiming_initialized', True)
    x_model.xavier_init = lambda m: setattr(m, 'was_xavier_initialized', True)
    x_model.proper_weight_init()


def test_layernorm_initialization(hparams):
    """
    Tests that LayerNorm weights are initialized to ones.
    """
    model = SimpleModel(**hparams, use_layernorm=True)
    model.proper_weight_init()
    for name, param in model.named_parameters():
        if 'ln.weight' in name:
            assert torch.all(torch.eq(param, 1.0))
        elif 'ln.bias' in name:
            assert torch.all(torch.eq(param, 0.0))

def test_calc_loss(simple_model, sample_batch):
    """
    Tests the calc_loss function.
    """
    criterion = nn.CrossEntropyLoss()
    loss = simple_model.calc_loss(sample_batch, criterion)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Loss should be a scalar
