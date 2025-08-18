
import torch.nn as nn
import torchsummary

from my_pytorch_kit.model.ae import ImageAE
from my_pytorch_kit.train.train import Trainer
from my_pytorch_kit.train.optimizers import get_optimizer_total_optimizer
from my_pytorch_kit.evaluation.reconstruction import ReconstructionEvaluator
from mnist.utils.mnist_utils import get_mnist_loaders
from mnist.autoencoder.ae_plots import plot_reconstructions 

if __name__ == '__main__':

    train = input("Train model? (y/n) ").lower() == "y"

    hparams = {
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 15,
        "optimizer_method": "Adam",
        # "optimizer_kwargs": {"weight_decay": 1e-4},
        "loss_func": nn.BCELoss(reduction="mean"),
        "feature_space": (64, 7, 7),
        "latent_dim": 4,
        "sample_input_shape": (1, 1, 28, 28),
    }

    model = ImageAE(**hparams)

    model.proper_weight_init()

    torchsummary.summary(model, (1, 28, 28))

    train_loader, val_loader, test_loader = get_mnist_loaders(hparams["batch_size"])

    optimizer = get_optimizer_total_optimizer(model, use_scheduler=False, use_grad_clip=False, **hparams)

    trainer = Trainer(model, train_loader, val_loader, **hparams)


    if train:

        load_model = input("Load model? (y/n) ").lower() == "y"

        if load_model:
            model.load_model("models/ae.pt")

        trainer.train(optimizer, **hparams)

        model.save_model("models/ae.pt")

    else:
        model.load_model("models/ae.pt")

    model.eval()


    evaluator = ReconstructionEvaluator(hparams["loss_func"])
    result = evaluator.evaluate(model, test_loader)

    print("Average loss on test set: ", result.item())

    plot_reconstructions(model, test_loader, plot_n= 15)
