import torch
from torch import nn
from torch.nn.functional import one_hot


class Generator(nn.Module):
    def __init__(
        self,
        output_size,
        hidden_size,
        latent_size,
        conditional=False,
        num_classes=10,
    ):
        super(Generator, self).__init__()
        # for vanilla GAN, no class information is required
        self.conditional = conditional
        self.num_classes = num_classes if conditional else 0

        self.fc1 = nn.Sequential(
            nn.Linear(latent_size + self.num_classes, hidden_size),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(hidden_size * 2, output_size), nn.Tanh()
        )

    def forward(self, z, y=None):
        z = z.view(z.size(0), -1)
        if self.conditional:
            device = next(self.parameters()).device
            y = one_hot(y.to(torch.int64), self.num_classes).to(self.device)
            z = torch.cat((z, y), dim=-1)
        out = self.fc4(self.fc3(self.fc2(self.fc1(z))))
        return out


class Discriminator(nn.Module):
    def __init__(
        self, input_size, hidden_size, conditional=False, num_classes=10
    ):
        super(Discriminator, self).__init__()
        self.conditional = conditional
        self.num_classes = num_classes if conditional else 0

        self.fc1 = nn.Sequential(
            nn.Linear(input_size + self.num_classes, hidden_size),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, y=None):
        x = x.reshape((x.size(0), -1))
        if self.conditional:
            device = next(self.parameters()).device
            y = one_hot(y.to(torch.int64), self.num_classes).to(device)
            x = torch.cat((x, y), dim=-1)
        out = self.fc4(self.fc3(self.fc2(self.fc1(x))))
        return out
