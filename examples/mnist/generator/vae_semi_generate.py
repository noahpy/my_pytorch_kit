
import torch
import matplotlib.pyplot as plt
import os

from my_pytorch_kit.model.vae import ImageVAESemiSupervised

def generate_samples(model, latent_dim, num_samples=20):
    """
    Generates num_samples samples from the model,
    plotting them in a grid with their labels
    """

    num_samples = int(num_samples)
    n_rows = num_samples // 10
    n_cols = 10

    model.eval()
    with torch.no_grad():
        z = torch.randn((num_samples, latent_dim))
        images, labels = model.generate(z)

        plt.figure(figsize=(10, 10))
        for i in range(num_samples):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(images[i].view(28, 28).cpu().numpy(), cmap="Greys_r")
            label = torch.argmax(labels[i]).item()
            plt.title(label)
            plt.axis("off")
        plt.show()


def generate_dataset(model, num_samples, latent_dim, std_divider=1.5, batch_size=32, path="data"):

    generated_count = 0

    path = f"{path}/vae_mnist"

    if not os.path.exists(path):
        os.makedirs(path)

    # create a folder for each class
    for i in range(10):
        os.makedirs(f"{path}/{i}")


    model.eval()
    with torch.no_grad():
        while generated_count < num_samples:

            next_batch_size = (num_samples - generated_count) % batch_size
            if next_batch_size == 0:
                next_batch_size = batch_size


            z = torch.randn((next_batch_size, latent_dim))
            z /= std_divider
            images, labels = model.generate(z)

            # save images to folder 
            for i in range(next_batch_size):
                plt.imsave(f"{path}/{torch.argmax(labels[i]).item()}/{generated_count}.png", images[i].view(28, 28).cpu().numpy(), cmap="Greys_r")
                generated_count += 1



if __name__ == '__main__':
    hparams = {
        "feature_space": (64, 7, 7),
        "latent_dim": 2,
        "sample_input_shape": (1, 1, 28, 28),
        "classifier_num_layers": 3
    }

    model = ImageVAESemiSupervised(**hparams)

    model.load_model("models/vae_semi_3.pt")

    generate_samples(model, hparams["latent_dim"])

    gen_dataset = input("Generate dataset? (y/[n]): ").lower() == "y"

    if gen_dataset:
        num_samples = int(input("Number of samples: "))
        generate_dataset(model, num_samples, hparams["latent_dim"], std_divider=0.8)
