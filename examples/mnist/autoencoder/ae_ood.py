
import torch
import torchvision
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

from my_pytorch_kit.model.ae import ImageAE
from mnist.utils.mnist_utils import get_mnist_loaders
from mnist.autoencoder.ae_plots import plot_reconstructions 
from my_pytorch_kit.evaluation.reconstruction import ReconstructionEvaluator

if __name__ == '__main__':

    model = ImageAE()
    model.load_model("models/ae.pt")

    vae_mnist_dataset = ImageFolder(
        root="data/vae_mnist",
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Grayscale()]),
    )

    vae_mnist_dataloader = torch.utils.data.DataLoader(vae_mnist_dataset, batch_size=32, shuffle=True, num_workers=2)

    mnist_train_loader, mnist_val_loader, mnist_test_loader = get_mnist_loaders(batch_size=32)

    sigmoid = torch.nn.Sigmoid()
    random_noise_loader = [sigmoid(torch.randn(32, 1, 28, 28)) for _ in range(1000)]

    evaluator = ReconstructionEvaluator(torch.nn.BCELoss(reduction="mean"), only_accumulate=True)

    print("Evaluating VAE MNIST...")
    vae_mnist_result = evaluator.evaluate(model, vae_mnist_dataloader)

    print("Evaluating MNIST...")
    mnist_result = evaluator.evaluate(model, mnist_test_loader)

    print("Evaluating random noise...")
    random_noise_result = evaluator.evaluate(model, random_noise_loader)


    fig, ax = plt.subplots()
    ax.boxplot([random_noise_result, vae_mnist_result, mnist_result])
    ax.set_xticklabels(["Random Noise", "VAE MNIST", "MNIST"])
    # ax.boxplot([vae_mnist_result, mnist_result])
    # ax.set_xticklabels(["VAE MNIST", "MNIST"])
    plt.show()




