
import torch
import torch.nn as nn
import torchsummary

from my_pytorch_kit.model.classifier import ImageClassifier
from my_pytorch_kit.train.train import Trainer
from my_pytorch_kit.train.optimizers import get_optimizer_total_optimizer
from mnist.utils.mnist_utils import get_mnist_loaders
from mnist.classifier.example import MnistAccEvaluator

class MnistAccEvaluator2(MnistAccEvaluator):

    def evaluate_batch(self, model, batch):
        model.use_softmax = True
        x, y = batch
        y_pred = model.forward(x)
        correct_prediction = (torch.argmax(y_pred, dim=1) == y).float()
        accuracy = correct_prediction.mean()
        model.use_softmax = False
        return accuracy


if __name__ == '__main__':

    train = input("Train model? (y/[n]) ").lower() == "y"

    hparams = {
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 5,
        "optimizer_method": "Adam",
        "optimizer_kwargs": {"weight_decay": 1e-4},
        "loss_func": nn.CrossEntropyLoss(),
        "feature_space": (64, 7, 7),
        "sample_input_shape": (1, 1, 28, 28),
    }

    model = ImageClassifier(**hparams)

    model.proper_weight_init()

    torchsummary.summary(model, (1, 28, 28))

    train_loader, val_loader, test_loader = get_mnist_loaders(hparams["batch_size"])

    optimizer = get_optimizer_total_optimizer(model, use_scheduler=False, use_grad_clip=False, **hparams)

    trainer = Trainer(model, train_loader, val_loader, **hparams)


    if train:

        load_model = input("Load model? (y/[n]) ").lower() == "y"

        if load_model:
            model.load_model("models/classifier.pt")

        trainer.train(optimizer, **hparams)

        model.save_model("models/classifier.pt")

    else:
        model.load_model("models/classifier.pt")

    model.eval()

    evaluator = MnistAccEvaluator2()
    result = evaluator.evaluate(model, test_loader)

    print(f"Model accuracy on test set: {result*100:.2f}%")
