import torch
from torch import nn
import numpy as np
from torchvision import datasets, transforms
import itertools
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


## Encoder
class Encoder(nn.Module):
    def __init__(self, EMBEDDING_DIM):
        super(Encoder, self).__init__()

        # Encoding layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, stride=2, kernel_size=3, bias=False, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, stride=1, kernel_size=3, bias=False, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, stride=2, kernel_size=3, bias=False, padding=1)
        self.linear1 = nn.Linear(in_features=64*161*3, out_features=128)  # Update if data dim changes
        self.linear2 = nn.Linear(in_features=128, out_features=EMBEDDING_DIM)

    def forward(self, x):
        _x = torch.relu(self.conv1(x))
        _x = torch.relu(self.conv2(_x))
        _x = torch.relu(self.conv3(_x))
        _x = torch.relu(torch.flatten(_x, 1))
        _x = torch.relu(self.linear1(_x))
        emb = self.linear2(_x)
        return emb


class Decoder(nn.Module):
    def __init__(self, EMBEDDING_DIM):
        super(Decoder, self).__init__()

        # Decoding layers
        self.linear3 = nn.Linear(in_features=EMBEDDING_DIM, out_features=128)
        self.linear4 = nn.Linear(in_features=128, out_features=64*161*3)  # Update if data dim changes
        self.convt1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, stride=1, kernel_size=3, padding=1, output_padding=0)
        self.convt2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, stride=2, kernel_size=3, padding=1, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(in_channels=16, out_channels=1, stride=2, kernel_size=3, padding=1, output_padding=1)

    def forward(self, emb):
        _x = torch.relu(self.linear3(emb))
        _x = torch.relu(self.linear4(_x))
        _x = _x.view(-1, 64, 161, 3)  # Update if data dim changes
        _x = torch.relu(self.convt1(_x))
        _x = torch.relu(self.convt2(_x))
        _x = torch.tanh(self.convt3(_x))
        return _x

class AutoEncoder(nn.Module):
    def __init__(self, EMBEDDING_DIM):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(EMBEDDING_DIM)
        self.decoder = Decoder(EMBEDDING_DIM)

    def forward(self, x):
        emb = self.encoder(x)
        _x = self.decoder(emb)
        return _x, emb
