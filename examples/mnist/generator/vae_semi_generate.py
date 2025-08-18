
import torch
import matplotlib.pyplot as plt

from my_pytorch_kit.model.vae import ImageVAESemiSupervised

def generate_samples(model, num_samples=20):
    """
    Generates num_samples samples from the model,
    plotting them in a grid with their labels
    """

    num_samples = int(num_samples)
    n_rows = num_samples // 10
    n_cols = 10

    model.eval()
    with torch.no_grad():
        z = torch.randn((num_samples, 2))
        images, labels = model.generate(z)

        plt.figure(figsize=(10, 10))
        for i in range(num_samples):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(images[i].view(28, 28).cpu().numpy(), cmap="Greys_r")
            label = torch.argmax(labels[i]).item()
            plt.title(label)
            plt.axis("off")
        plt.show()
        




if __name__ == '__main__':
    hparams = {
        "feature_space": (64, 7, 7),
        "latent_dim": 2,
        "sample_input_shape": (1, 1, 28, 28),
        "classifier_num_layers": 3
    }

    model = ImageVAESemiSupervised(**hparams)

    model.load_model("models/vae.pt")

    generate_samples(model)


