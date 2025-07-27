import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Assuming the provided code is in these files
from my_pytorch_kit.train.train import train_model
from my_pytorch_kit.train.optimizers import TotalOptimizer
from my_pytorch_kit.model.models import BaseModel


# --- Test-Specific Mock Classes ---

# Create a dummy class that IS a subclass of BaseModel to satisfy the type check
# We assume BaseModel can be instantiated without arguments.
# If it requires arguments, they would be passed here.
class MockModelForTest(BaseModel):
    def __init__(self):
        super().__init__({})
        # If BaseModel is a torch.nn.Module, it's good practice to have a parameter
        self._param = torch.nn.Parameter(torch.empty(1))

    def forward(self, *args, **kwargs):
        pass # Not needed for this test

# --- Pytest Fixtures ---

@pytest.fixture
def mock_model():
    """
    Fixture for a mock model that is a true instance of a BaseModel subclass.
    """
    # 1. Instantiate our test-specific model class
    model_instance = MockModelForTest()

    # 2. Attach MagicMocks for the methods we need to control and assert on.
    # This gives us a real object that passes type checks, but with mock methods.
    model_instance.train = MagicMock()
    model_instance.eval = MagicMock()
    model_instance.calc_loss = MagicMock(return_value=torch.tensor(0.5, requires_grad=True))
    # Mock parameters() as it's needed by the optimizer
    model_instance.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(1))])

    return model_instance


@pytest.fixture
def mock_optimizer():
    """
    Fixture for a mock TotalOptimizer. It explicitly mocks the nested
    optimizer structure to avoid AttributeErrors.
    It uses `spec=TotalOptimizer` so `isinstance(mock, TotalOptimizer)` passes.
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
    return [(torch.randn(2, 3), torch.randn(2, 1))] * 5


# --- Tests ---

def test_train_model_happy_path(mock_model, mock_optimizer, mock_loss_func, mock_tb_logger, mock_data_loader):
    """
    Tests the successful execution of the train_model function.
    """
    hparams = {"epochs": 2, "loss_cutoff_rate": 0.1}

    # This will now pass the issubclass check
    train_model(mock_model, hparams, mock_data_loader, mock_data_loader,
                mock_loss_func, mock_optimizer, mock_tb_logger)

    # Assertions for training loop
    assert mock_model.train.call_count == hparams["epochs"]
    assert mock_optimizer.zero_grad.call_count == hparams["epochs"] * len(mock_data_loader)
    assert mock_model.calc_loss.call_count == hparams["epochs"] * len(mock_data_loader) * 2
    assert mock_optimizer.step.call_count == hparams["epochs"] * len(mock_data_loader)

    # Assertions for validation loop
    assert mock_model.eval.call_count == hparams["epochs"]

    # Assertions for tensorboard logging
    assert mock_tb_logger.add_scalar.call_count == hparams["epochs"] * (len(mock_data_loader) + len(mock_data_loader))


def test_train_model_keyboard_interrupt(mock_model, mock_optimizer, mock_loss_func, mock_tb_logger, mock_data_loader):
    """
    Tests that the training gracefully stops on a KeyboardInterrupt.
    """
    hparams = {"epochs": 5}

    with patch('my_pytorch_kit.train.train.create_tqdm_bar', side_effect=KeyboardInterrupt):
        train_model(mock_model, hparams, mock_data_loader, mock_data_loader,
                    mock_loss_func, mock_optimizer, mock_tb_logger)

        assert mock_model.train.call_count <= 1


def test_train_model_invalid_model_type(mock_optimizer, mock_loss_func, mock_tb_logger, mock_data_loader):
    """
    Tests that a ValueError is raised for a model that is not a subclass of BaseModel.
    """
    hparams = {"epochs": 1}
    # A generic mock will fail the 'issubclass' check, which is what we want to test here
    invalid_model = MagicMock()

    with pytest.raises(ValueError, match="Model with type .* must be a subclass of BaseModel"):
        train_model(invalid_model, hparams, mock_data_loader, mock_data_loader,
                    mock_loss_func, mock_optimizer, mock_tb_logger)


def test_train_model_invalid_optimizer_type(mock_model, mock_loss_func, mock_tb_logger, mock_data_loader):
    """
    Tests that a ValueError is raised for an optimizer that is not an instance of TotalOptimizer.
    """
    hparams = {"epochs": 1}
    # A generic mock will fail the 'isinstance' check for TotalOptimizer
    invalid_optimizer = MagicMock()

    with pytest.raises(ValueError, match="Optimizer must be an instance of TotalOptimizer"):
        train_model(mock_model, hparams, mock_data_loader, mock_data_loader,
                    mock_loss_func, invalid_optimizer, mock_tb_logger)

def test_train_model_override_instance_errors(mock_loss_func, mock_tb_logger, mock_data_loader):
    """
    Tests that the instance type checks can be overridden.
    """
    hparams = {"epochs": 1}
    # Use generic mocks that would normally fail the type checks
    invalid_model = MagicMock()
    invalid_model.calc_loss.return_value = torch.tensor(0.5, requires_grad=True)

    torch_optimizer_mock = MagicMock()
    torch_optimizer_mock.param_groups = [{'lr': 0.001}]
    invalid_optimizer = MagicMock()
    invalid_optimizer.optimizer = torch_optimizer_mock

    # This should not raise a ValueError because of the override flag
    train_model(invalid_model, hparams, mock_data_loader, mock_data_loader,
                mock_loss_func, invalid_optimizer, mock_tb_logger, override_instance_errors=True)

    # Assert that the training still proceeded for one epoch
    assert invalid_model.train.call_count == 1
    assert invalid_optimizer.zero_grad.call_count == len(mock_data_loader)
