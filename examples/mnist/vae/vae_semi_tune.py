
import torch
import torch.nn as nn
import tqdm

from my_pytorch_kit.model.vae import ImageVAESemiSupervised
from my_pytorch_kit.train.tune import Tuner
from my_pytorch_kit.evaluation.evaluation import Evaluator 
from my_pytorch_kit.model.classifier import ImageClassifier
from my_pytorch_kit.model.ae import ImageAE
from mnist.utils.mnist_utils import get_mnist_loaders
from mnist.autoencoder.ae_plots import plot_latent, plot_reconstructions, plot_label_clusters


class GenerationEvaluator(Evaluator):


    def __init__(self,
                 latent_dim = 2,
                 num_samples = 1000,
                 batch_size = 64,
                 ce_weight = 1,
                 bce_weight = 2.5):
        super().__init__()
        self.batch_count = 0
        self.result = 0

        self.classifier = ImageClassifier()
        self.classifier.load_model("models/classifier.pt")

        self.ae = ImageAE()
        self.ae.load_model("models/ae.pt")

        self.latent_dim = latent_dim
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.bce = nn.BCELoss(reduction="mean")
        self.ce = nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.1)

        self.ce_weight = ce_weight
        self.bce_weight = bce_weight

    def on_eval(self, model):

        print("\nEvaluating...")
        generated_count = 0

        acc = 0

        model.eval()
        pbar = tqdm.tqdm(total=self.num_samples // self.batch_size + 1)
        with torch.no_grad():
            while generated_count < self.num_samples:

                next_batch_size = (self.num_samples - generated_count) % self.batch_size
                if next_batch_size == 0:
                    next_batch_size = self.batch_size


                z = torch.randn((next_batch_size, self.latent_dim))
                images, labels = model.generate(z)

                cl_labels = self.classifier(images)
                cl_labels = torch.argmax(cl_labels, dim=1)

                ce_loss = self.ce(labels, cl_labels)


                reconstruction = self.ae(images)
                recon_losses = self.bce(reconstruction, images)

                loss = self.bce_weight * recon_losses.item() + self.ce_weight * ce_loss.item()

                acc += loss

                generated_count += next_batch_size

                pbar.update(1)
        self.result = acc

    def evaluate_batch(self, model, batch):
        return 0

    def accumulate_result(self, result):
        return

    def get_result(self):
        return self.result

def kl_annealing(epoch, epochs, model, **kwargs):
    model.beta = model.beta + (kwargs["final_beta"] - kwargs["start_beta"]) / (epochs - kwargs["beta_start_delay"])


if __name__ == '__main__':

    hparams = {
        "learning_rate": ((1e-4, 1e-2), "log"),
        "batch_size": 64,
        "epochs": 10,
        "patience": 5,
        "optimizer_method": "Adam",
        # "optimizer_kwargs": {"weight_decay": 1e-4},
        "loss_func": nn.BCELoss(reduction="sum"),
        "alpha": ((0, 10), "float"),
        "beta": ((0, 10), "float"),
        "classifier_loss_weight": ((0, 10), "float"),
        "start_beta": ((0, 1), "float"),
        "final_beta": ((1, 10), "float"),
        "beta_start_delay": ((4, 12), "int"),
        "feature_space": ([(64, 7, 7)], "item"),
        "latent_dim": 2,
        "sample_input_shape": ([(1, 1, 28, 28)], "item"),
        "classifier_num_layers": 3,
        "use_scheduler": ((True, False, False), "item"),
        "use_grad_clip": ((True, False, False), "item"),
        "gamma": ((0.8, 0.95), "log"),
        "epoch_function": ((kl_annealing, None, None), "item"),
    }


    train_loader, val_loader, test_loader = get_mnist_loaders(hparams["batch_size"])

    evaluator = GenerationEvaluator()

    tuner = Tuner(ImageVAESemiSupervised)

    num_search = 500
    ranks_considered = 15
    check_multiplier = 3

    # train model
    model, best_config, results = tuner.tune(train_loader, val_loader, hparams,
                                             evaluator=evaluator,
                                             num_search=num_search,
                                             ranks_considered=ranks_considered,
                                             check_multiplicant=check_multiplier,
                                             mode='dynamic', model_path='models/mnist_vae_semi_tuned_')

    model.eval()

    model.save_model("models/mnist_vae_semi_tuned_best.pt", best_config)

    plot_reconstructions(model, test_loader)

    distribution = plot_label_clusters(model.encoder, test_loader)

    plot_latent(model.decoder, dist=distribution, stds=2)
