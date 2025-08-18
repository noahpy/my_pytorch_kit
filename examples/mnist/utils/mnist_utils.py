
from torchvision.datasets import MNIST
import torchvision
import torch

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
        mnist_test_dataset, batch_size=batch_size, shuffle=True
    )

    return train_loader, val_loader, test_loader

