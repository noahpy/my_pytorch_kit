
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

def plot_latent(decoder, dist=None, n = 15, scale = 2, img_dim = 28, figsize = 15, stds = 3):
    # display a n * n 2D manifold of images
    figure = np.zeros((img_dim * n, img_dim * n))

    x_space = [-scale, scale]
    y_space = [-scale, scale]

    if dist:
        x_space = [dist[0][0] - stds * dist[0][1], dist[0][0] + stds * dist[0][1]]
        y_space = [dist[1][0] - stds * dist[1][1], dist[1][0] + stds * dist[1][1]]

    # linearly spaced coordinates corresponding to the 2D plot
    # of images classes in the latent space
    grid_x = np.linspace(*x_space, n)
    grid_y = np.linspace(*y_space, n)[::-1]

    sigmoid = nn.Sigmoid()

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]]).float()
            x_decoded = decoder(z_sample)
            x_decoded = sigmoid(x_decoded)
            images = x_decoded.reshape(img_dim, img_dim).detach().numpy()
            figure[
                i * img_dim: (i + 1) * img_dim,
                j * img_dim: (j + 1) * img_dim,
            ] = images

    plt.figure(figsize =(figsize, figsize))
    start_range = img_dim // 2
    end_range = n * img_dim + start_range + 1
    pixel_range = np.arange(start_range, end_range, img_dim)[:-1]
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)

    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap ="Greys_r")
    plt.show()


def plot_reconstructions(model, loader, plot_n=10):
    model.eval()
    plotted = 0
    for batch_idx, (data, _) in enumerate(loader):
        with torch.no_grad():
            recon_batch = model(data)
            if isinstance(recon_batch, tuple):
                recon_batch = recon_batch[0]
            n = data.size(0)
            for i in range(n):
                plt.subplot(2, plot_n, plotted + 1)
                plt.imshow(data[i].view(28, 28).cpu().numpy(), cmap="Greys_r")
                plt.subplot(2, plot_n, plotted + 1 + plot_n)
                plt.imshow(recon_batch[i].view(28, 28).cpu().numpy(), cmap="Greys_r")
                plotted += 1
                if plotted >= plot_n:
                    break
        if plotted >= plot_n:
            break
    plt.show()

def plot_label_clusters(encoder, loader, batch_num=100, on_logvar=False):

    encoder.eval()

    total_z_mean = torch.tensor([])
    total_z_logvar = torch.tensor([])
    total_labels = torch.tensor([])

    for i, batch in enumerate(loader):
        data, labels = batch
        out = encoder(data)
        z_mean, z_logvar = out.split(2, dim = -1)
        total_z_mean = torch.concat((total_z_mean, z_mean), axis = 0)
        total_z_logvar = torch.concat((total_z_logvar, z_logvar), axis = 0)
        total_labels = torch.concat((total_labels, labels), axis = 0)

        if i >= batch_num:
            break


    total_z_mean = total_z_mean.detach().numpy()
    total_z_logvar = total_z_logvar.detach().numpy()
    total_labels = total_labels.detach().numpy()

    z0_mean = np.mean(total_z_mean[:, 0])
    z0_std = np.std(total_z_mean[:, 0])
    z1_mean = np.mean(total_z_mean[:, 1])
    z1_std = np.std(total_z_mean[:, 1])


    if on_logvar:
        total_z_mean = total_z_logvar

    plt.figure(figsize =(12, 10))
    sc = plt.scatter(total_z_mean[:, 0], total_z_mean[:, 1], c = total_labels)
    cbar = plt.colorbar(sc, ticks = range(10))
    cbar.ax.set_yticklabels([i for i in range(10)])
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()

    return [[z0_mean, z0_std], [z1_mean, z1_std]]
