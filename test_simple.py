from torchmetrics.functional import peak_signal_noise_ratio as psnr
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from autoencoder import Encoder, Decoder, AutoEncoderMRL
from imagedataset import ImageDataset
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.optim import Adam
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
device = torch.device('cuda')


autoencoder_mrl = AutoEncoderMRL().to(device)
autoencoder_mrl = torch.load("update_ae_50.pth")
autoencoder_mrl.eval()
total_loss = 0

transform = transforms.Compose([
    transforms.ToTensor(),          # Convert image to PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

dataset_test_512 = ImageDataset("data/test/512", transform=transform)
dataset_test_256 = ImageDataset("data/test/256", transform=transform) 
dataset_test_128 = ImageDataset("data/test/128", transform=transform) 
batch_size = 32
dataloader_test_512 = DataLoader(dataset_test_512, batch_size=batch_size, shuffle=False)
dataloader_test_256 = DataLoader(dataset_test_256, batch_size=batch_size, shuffle=False)
dataloader_test_128 = DataLoader(dataset_test_128, batch_size=batch_size, shuffle=False)

mrl_weights = [1,1,1]
total_loss = 0

autoencoder_simple = AutoEncoderMRL().to(device)
autoencoder_simple = torch.load("simple_update_ae_20.pth")
autoencoder_simple.eval()
total_loss = 0
def psnr_loss(output, target):
    mse = F.mse_loss(output, target, reduction='mean')
    psnr = -10 * torch.log10(mse + 1e-8)  # Adding small epsilon to avoid log(0)
    return -psnr  # Negative for minimization
dataset_test_512 = ImageDataset("data/test/512", transform=transform)

batch_size = 32
dataloader_test_512 = DataLoader(dataset_test_512, batch_size=batch_size, shuffle=False)

with torch.no_grad():
    for images_512 in dataloader_test_512:
        images_512 = images_512.to(device)  # 512x512 images

        _, _, out_512, _ = autoencoder_simple(images_512)

        loss = -psnr(out_512, images_512,data_range=1.0)

        total_loss += loss.item()

avg_loss = total_loss / len(dataloader_test_512)
print(f"Total Loss: {avg_loss:.4f}")
