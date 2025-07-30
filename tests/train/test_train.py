import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Assuming the refactored code is in these files
from my_pytorch_kit.train.train import Trainer
from my_pytorch_kit.train.optimizers import TotalOptimizer
from my_pytorch_kit.model.models import BaseModel


# --- Test-Specific Mock Classes ---

# Create a dummy class that IS a subclass of BaseModel to satisfy the type check
class MockModelForTest(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        # A parameter is often needed for optimizers to initialize
        self._param = torch.nn.Parameter(torch.empty(1))

    def forward(self, *args, **kwargs):
        # A forward pass method is required by nn.Module, but its logic isn't tested here.
        pass

# --- Pytest Fixtures ---

@pytest.fixture
def mock_model():
    """
    Fixture for a mock model that is a true instance of a BaseModel subclass.
    Its methods are replaced with MagicMocks for tracking calls.
    """
    model_instance = MockModelForTest()
    model_instance.train = MagicMock()
    model_instance.eval = MagicMock()
    model_instance.calc_loss = MagicMock(return_value=torch.tensor(0.5, requires_grad=True))
    model_instance.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(1))])
    return model_instance

@pytest.fixture
def mock_optimizer():
    """
    Fixture for a mock TotalOptimizer. It explicitly mocks the nested
    optimizer structure to avoid AttributeErrors.
    """
    torch_optimizer_mock = MagicMock(spec=torch.optim.Optimizer)
    torch_optimizer_mock.param_groups = [{'lr': 0.001}]

    total_optimizer_mock = MagicMock(spec=TotalOptimizer)
    total_optimizer_mock.optimizer = torch_optimizer_mock

    return total_optimizer_mock

@pytest.fixture
def mock_loss_func():
    """Fixture for a mock loss function."""
    return MagicMock()

@pytest.fixture
def mock_tb_logger():
    """Fixture for a mock tensorboard logger."""
    return MagicMock()

@pytest.fixture
def mock_data_loader():
    """Fixture for a mock data loader."""
    # An iterable of tuples, mimicking a real DataLoader
    return [(torch.randn(2, 3), torch.randn(2, 1))] * 5


# --- Tests ---

def test_trainer_initialization_with_default_logger(mock_model, mock_data_loader):
    """
    Tests that get_tensorboard_logger is called when a logger is not provided.
    """
    with patch('my_pytorch_kit.train.train.get_tensorboard_logger') as mock_get_logger:
        # Initialize the trainer without a tb_logger
        trainer = Trainer(mock_model, mock_data_loader, mock_data_loader, tb_logger=None)
        
        # Assert that our fallback function was called exactly once
        mock_get_logger.assert_called_once()
        # Assert that the trainer's logger is the one returned by the mocked function
        assert trainer.tb_logger == mock_get_logger.return_value

def test_train_happy_path(mock_model, mock_optimizer, mock_loss_func, mock_tb_logger, mock_data_loader):
    """
    Tests the successful execution of the train method.
    """
    hparams = {"epochs": 2, "loss_cutoff_rate": 0.1}
    
    # 1. Setup: Instantiate the Trainer
    trainer = Trainer(mock_model, mock_data_loader, mock_data_loader, mock_tb_logger)
    
    # 2. Action: Run the training
    trainer.train(mock_loss_func, mock_optimizer, **hparams)

    # 3. Assertions
    assert mock_model.train.call_count == hparams["epochs"]
    assert mock_model.eval.call_count == hparams["epochs"]
    assert mock_optimizer.zero_grad.call_count == hparams["epochs"] * len(mock_data_loader)
    assert mock_optimizer.step.call_count == hparams["epochs"] * len(mock_data_loader)
    # calc_loss is called for both training and validation loops
    assert mock_model.calc_loss.call_count == hparams["epochs"] * len(mock_data_loader) * 2
    # add_scalar is called for each batch in training and validation
    assert mock_tb_logger.add_scalar.call_count == hparams["epochs"] * len(mock_data_loader) * 2

def test_train_keyboard_interrupt(mock_model, mock_optimizer, mock_loss_func, mock_tb_logger, mock_data_loader):
    """
    Tests that the training gracefully stops on a KeyboardInterrupt.
    """
    hparams = {"epochs": 5}
    trainer = Trainer(mock_model, mock_data_loader, mock_data_loader, mock_tb_logger)

    with patch('my_pytorch_kit.train.train.create_tqdm_bar', side_effect=KeyboardInterrupt):
        trainer.train(mock_loss_func, mock_optimizer, **hparams)
        # Assert that the training started but did not complete all epochs
        assert mock_model.train.call_count == 1
        assert mock_model.eval.call_count == 0 # Should exit before validation

def test_train_invalid_model_type(mock_optimizer, mock_loss_func, mock_tb_logger, mock_data_loader):
    """
    Tests that a ValueError is raised for a model that is not a subclass of BaseModel.
    """
    hparams = {"epochs": 1}
    invalid_model = MagicMock() # This mock will fail the 'issubclass' check
    
    trainer = Trainer(invalid_model, mock_data_loader, mock_data_loader, mock_tb_logger)

    with pytest.raises(ValueError, match="model with type .* must be a subclass of BaseModel"):
        trainer.train(mock_loss_func, mock_optimizer, **hparams)

def test_train_invalid_optimizer_type(mock_model, mock_loss_func, mock_tb_logger, mock_data_loader):
    """
    Tests that a ValueError is raised for an optimizer that is not an instance of TotalOptimizer.
    """
    hparams = {"epochs": 1}
    invalid_optimizer = MagicMock() # This mock will fail the 'isinstance' check

    trainer = Trainer(mock_model, mock_data_loader, mock_data_loader, mock_tb_logger)

    with pytest.raises(ValueError, match="Optimizer must be an instance of TotalOptimizer"):
        trainer.train(mock_loss_func, invalid_optimizer, **hparams)

def test_train_override_instance_errors(mock_loss_func, mock_tb_logger, mock_data_loader):
    """
    Tests that the instance type checks can be overridden.
    """
    hparams = {"epochs": 1}
    # Setup invalid mocks that would normally fail type checks
    invalid_model = MagicMock()
    invalid_model.calc_loss.return_value = torch.tensor(0.5, requires_grad=True)
    
    torch_optimizer_mock = MagicMock()
    torch_optimizer_mock.param_groups = [{'lr': 0.001}]
    invalid_optimizer = MagicMock()
    invalid_optimizer.optimizer = torch_optimizer_mock
    invalid_optimizer.param_groups = [{'lr': 0.001}]

    trainer = Trainer(invalid_model, mock_data_loader, mock_data_loader, mock_tb_logger)

    # This should not raise a ValueError because of the override flag
    trainer.train(mock_loss_func, invalid_optimizer, override_instance_errors=True, **hparams)

    # Assert that the training still proceeded for one epoch
    assert invalid_model.train.call_count == 1
    assert invalid_optimizer.zero_grad.call_count == len(mock_data_loader)
