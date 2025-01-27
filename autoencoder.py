import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)  # 512 -> 256
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)  # 256 -> 128
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)  # 128 -> 64
        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)  # 64 -> 32
        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(64 * 64 * 64, 1024)  # Fully connected layer
        # self.fc2 = nn.Linear(1024, 256)  # Fully connected layer

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # 512 -> 256
        x = torch.relu(self.conv2(x))  # 256 -> 128
        # x = torch.relu(self.conv3(x))  # 128 -> 64
        # x = torch.relu(self.conv4(x))  # 64 -> 32
        # x = self.flatten(x)
        # x = torch.relu(self.fc1(x))  # Fully connected layer
        # x = torch.relu(self.fc2(x))  # Fully connected layer
        return x  # Final compressed representation


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # self.fc1 = nn.Linear(256, 1024)
        # self.fc2 = nn.Linear(1024, 64 * 64 * 64)
        # self.unflatten = nn.Unflatten(1, (64, 64, 64))
        # self.convt1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)  # 32 -> 64
        # self.convt2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)  # 64 -> 128
        self.convt3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)  # 128 -> 256
        self.convt4 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)  # 256 -> 512

        # Additional outputs for intermediate resolutions
        self.out_128 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)  # Output at 128x128
        self.out_256 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)  # Output at 256x256

    def forward(self, x):
        # x = torch.relu(self.fc1(x))  # Fully connected layer
        # x = torch.relu(self.fc2(x))  # Fully connected layer
        # x = self.unflatten(x)
        # # x = torch.relu(self.convt1(x))  # 32 -> 64
        # x = torch.relu(self.convt2(x))  # 64 -> 128

        out_128 = torch.tanh(self.out_128(x))  # 128x128 output

        x = torch.relu(self.convt3(x))  # 128 -> 256
        out_256 = torch.tanh(self.out_256(x))  # 256x256 output

        x = torch.relu(self.convt4(x))  # 256 -> 512
        out_512 = torch.tanh(x)  # 512x512 output

        return out_128, out_256, out_512


class AutoEncoderMRL(nn.Module):
    def __init__(self):
        super(AutoEncoderMRL, self).__init__()
        self.encoder = Encoder()  # Downsampling: 512 -> 32
        self.decoder = Decoder()  # Upsampling: 32 -> 512

    def forward(self, x):
        emb = self.encoder(x)  # Get compressed representation
        out_128, out_256, out_512 = self.decoder(emb)  # Reconstruct at multiple resolutions
        return out_128, out_256, out_512, emb
