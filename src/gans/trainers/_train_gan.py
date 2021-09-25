import torch
from torch import nn
import logging
from tqdm import trange

from ..models.GAN import Generator, Discriminator


logger = logging.getLogger(__name__)


def train_gan(
    dataloader: torch.utils.data.DataLoader,
    batch_size: int = 100,
    input_size: int = 784,
    hidden_size: int = 512,
    latent_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 0.0002,
    weight_decay: float = 0,
    device: torch.cuda.device = None,
) -> torch.nn.Module:
    G = Generator(input_size, hidden_size, latent_size).to(device)
    D = Discriminator(input_size, hidden_size).to(device)

    criterion = nn.BCELoss()
    g_optimizer = torch.optim.Adam(
        params=G.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    d_optimizer = torch.optim.Adam(
        params=D.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # start training
    for epoch in trange(num_epochs):
        for images, _ in dataloader:
            images = images.to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # ========== TRAIN THE DISCRIMINATOR ==========
            # compute loss using real images and real labels
            pred_real = D(images)
            d_loss_real = criterion(pred_real, real_labels)

            # compute loss using fake images and fake labels
            z = torch.randn(batch_size, latent_size).to(device)
            fake_images = G(z)
            pred_fake = D(fake_images)
            d_loss_fake = criterion(pred_fake, fake_labels)

            # b-p to update discriminator
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # ========== TRAIN THE GENERATOR ==========
            # compute loss using fake images and real labels
            z = torch.randn(batch_size, latent_size).to(device)
            fake_images = G(z)
            pred_fake = D(fake_images)
            g_loss = criterion(pred_fake, real_labels)

            # b-p to update generator
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        # print results for the last batch after every some epochs
        if (epoch + 1) % 50 == 0:
            logger.info(
                "Epoch [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}".format(
                    epoch + 1, num_epochs, d_loss, g_loss
                )
            )
    return G
