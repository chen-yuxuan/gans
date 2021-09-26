import torch
from torch import nn
from torchvision.utils import save_image
import numpy as np
import logging
from tqdm import trange

from ..models.GAN import Generator, Discriminator


logger = logging.getLogger(__name__)


def train_cgan(
    dataloader: torch.utils.data.DataLoader,
    batch_size: int = 100,
    input_shape: torch.Size = torch.Size([28, 28]),
    num_classes: int = 10,
    hidden_size: int = 512,
    latent_size: int = 16,
    num_epochs: int = 100,
    g_learning_rate: float = 0.0001,
    d_learning_rate: float = 0.0001,
    weight_decay: float = 0,
    device: torch.cuda.device = None,
) -> torch.nn.Module:
    logger.info(locals())

    input_size = int(np.prod(input_shape))
    G = Generator(
        input_size, hidden_size, latent_size, conditional=True, num_classes=num_classes
    ).to(device)
    D = Discriminator(
        input_size, hidden_size, conditional=True, num_classes=num_classes
    ).to(device)

    criterion = nn.BCELoss()
    g_optimizer = torch.optim.Adam(
        params=G.parameters(), lr=g_learning_rate, weight_decay=weight_decay
    )
    d_optimizer = torch.optim.Adam(
        params=D.parameters(), lr=d_learning_rate, weight_decay=weight_decay
    )

    # start training
    tbar = trange(num_epochs, leave=True)
    for epoch in tbar:
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # ========== TRAIN THE GENERATOR ==========
            g_optimizer.zero_grad()
            # compute loss using fake images and real labels
            z = torch.randn(batch_size, latent_size).to(device)
            fake_images = G(z, labels)
            pred_fake = D(fake_images, labels)
            g_loss = criterion(pred_fake, real_labels)

            # b-p to update generator
            g_loss.backward()
            g_optimizer.step()

            # ========== TRAIN THE DISCRIMINATOR ==========
            d_optimizer.zero_grad()
            # compute loss using real images and real labels
            pred_real = D(images, labels)
            d_loss_real = criterion(pred_real, real_labels)

            # compute loss using fake images and fake labels
            z = torch.randn(batch_size, latent_size).to(device)
            fake_images = G(z, labels)
            pred_fake = D(fake_images, labels)
            d_loss_fake = criterion(pred_fake, fake_labels)

            # b-p to update discriminator
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

        # update the loss of the last batch for each epoch
        tbar.set_postfix(
            {"d_loss": round(d_loss.item(), 3), "g_loss": round(g_loss.item(), 3)}
        )
        # save generated images every 10 epochs
        if (epoch + 1) % 10 == 0:
            generated_images = fake_images[:64].view(
                -1, 1, input_shape[-2], input_shape[-1]
            )
            # save images
            save_image(
                generated_images,
                "./{}.png".format(epoch + 1),
                nrow=8,
                normalize=True,
            )
    return G
