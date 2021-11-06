import torch
from torchvision.utils import save_image
import numpy as np
import logging
from tqdm import trange

from ..models.GAN import Generator, Discriminator


logger = logging.getLogger(__name__)

CLIP_WEIGHT = 0.01
N_CRITIC = 5


def train_wgan(
    dataloader: torch.utils.data.DataLoader,
    batch_size: int = 100,
    input_shape: torch.Size = torch.Size([28, 28]),
    hidden_size: int = 256,
    latent_size: int = 64,
    num_epochs: int = 150,
    g_lr: float = 0.00005,
    d_lr: float = 0.00005,
    weight_decay: float = 0,
    device: torch.cuda.device = None,
) -> torch.nn.Module:
    logger.info(locals())

    # unpack input shape
    input_size = int(np.prod(input_shape))
    if len(input_shape) == 3:
        num_channels, height, width = input_shape
    elif len(input_shape) == 2:
        num_channels = 1
        height, width = input_shape

    # set model
    G = Generator(input_size, hidden_size, latent_size).to(device)
    D = Discriminator(input_size, hidden_size).to(device)

    # set optimizers
    g_optimizer = torch.optim.RMSprop(
        params=G.parameters(), lr=g_lr, weight_decay=weight_decay
    )
    d_optimizer = torch.optim.RMSprop(
        params=D.parameters(), lr=d_lr, weight_decay=weight_decay
    )

    # start training
    tbar = trange(num_epochs, leave=True)
    for epoch in tbar:
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)

            # ========== TRAIN THE DISCRIMINATOR ==========
            d_optimizer.zero_grad()
            z = torch.randn(batch_size, latent_size).to(device)
            # compute loss using distribution of D predictions
            # The closer `d_loss` -> 0, the better
            pred_real = D(images)
            d_loss_real = -torch.mean(pred_real)

            fake_images = G(z)
            pred_fake = D(fake_images)
            d_loss_fake = torch.mean(pred_fake)

            # b-p to update discriminator
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # ========== CLIP DISCRIMINATOR WEIGHTS ==========
            for weight in D.parameters():
                weight.data.clamp_(-CLIP_WEIGHT, CLIP_WEIGHT)

            # ========== TRAIN THE GENERATOR ==========
            if (i + 1) % N_CRITIC == 0:
                g_optimizer.zero_grad()
                z = torch.randn(batch_size, latent_size).to(device)
                # compute loss using the distribution of D predictions
                # The closer `g_loss` -> -0.5, the better
                fake_images = G(z)
                pred_fake = D(fake_images)
                g_loss = -torch.mean(pred_fake)

                # b-p to update generator
                g_loss.backward()
                g_optimizer.step()

        # update the loss of the last batch for each epoch
        tbar.set_postfix(
            {"d_loss": round(d_loss.item(), 3), "g_loss": round(g_loss.item(), 3)}
        )
        # save generated images every 10 epochs
        if (epoch + 1) % 10 == 0:
            generated_images = fake_images[:64].view(-1, num_channels, height, width)
            # save images
            save_image(
                generated_images,
                "./{}.png".format(epoch + 1),
                nrow=8,
                normalize=True,
            )
    return G
