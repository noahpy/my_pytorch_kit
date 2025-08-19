
import torch.nn as nn
import torchsummary

from my_pytorch_kit.model.vae import ImageVAE
from my_pytorch_kit.model.vae import ImageVAESemiSupervised
from my_pytorch_kit.train.train import Trainer
from my_pytorch_kit.train.optimizers import get_optimizer_total_optimizer
from mnist.utils.mnist_utils import get_mnist_loaders
from mnist.autoencoder.ae_plots import plot_latent, plot_reconstructions, plot_label_clusters


def kl_annealing(epoch, epochs, model, **kwargs):
    model.beta = model.beta + (kwargs["final_beta"] - kwargs["start_beta"]) / epochs
    print(f"Set beta to: {model.beta}")


if __name__ == '__main__':

    train = input("Train new model? (y/[n]) ").lower() == "y"

    hparams = {
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 15,
        "optimizer_method": "Adam",
        # "optimizer_kwargs": {"weight_decay": 1e-4},
        "loss_func": nn.BCELoss(reduction="sum"),
        "alpha": 1e-1,
        "beta": 1.5e-3,
        "classifier_loss_weight": 1.2,
        "start_beta": 1e-3,
        "final_beta": 4e-2,
        "feature_space": (64, 7, 7),
        "latent_dim": 4,
        "sample_input_shape": (1, 1, 28, 28),
        "classifier_num_layers": 3,
        # "epoch_function": kl_annealing
    }

    model = ImageVAESemiSupervised(**hparams)
    # model = ImageVAE(**hparams)

    torchsummary.summary(model, (1, 28, 28))

    train_loader, val_loader, test_loader = get_mnist_loaders(hparams["batch_size"])

    optimizer = get_optimizer_total_optimizer(model, use_scheduler=False, use_grad_clip=False, **hparams)

    trainer = Trainer(model, train_loader, val_loader, **hparams)


    if train:
        load_model = input("Load model? (y/[n]) ").lower() == "y"

        if load_model:
            path = input("Enter path to model (default: models/vae.pt) :")
            if path == "":
                path = "models/vae.pt"
            model.load_model(path)

        trainer.train(optimizer, **hparams)

        model.save_model("models/vae.pt", hparams)

    else:
        model.load_model("models/vae.pt")

    model.eval()

    plot_reconstructions(model, test_loader)

    distribution = plot_label_clusters(model.encoder, test_loader)

    plot_latent(model.decoder, dist=distribution, stds=2)


