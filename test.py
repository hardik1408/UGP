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
def mrl_loss(outputs, target_128, target_256, target_512, weights):
    out_128, out_256, out_512 = outputs
    loss_128 = -psnr(out_128, target_128, data_range=1.0)
    loss_256 = -psnr(out_256, target_256, data_range=1.0)
    loss_512 = -psnr(out_512, target_512, data_range=1.0)

    # print(f"PSNR Loss at 128x128: {-loss_128:.4f}, 256x256: {-loss_256:.4f}, 512x512: {-loss_512:.4f}")

    total_loss = weights[0] * loss_128 + weights[1] * loss_256 + weights[2] * loss_512
    return loss_128 , loss_256 , loss_512 , total_loss / 3 


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
total_loss_128 = 0
total_loss_256 = 0
total_loss_512 = 0
with torch.no_grad():
    for batch_idx, (images_512, images_256, images_128) in enumerate(zip(dataloader_test_512, dataloader_test_256, dataloader_test_128)):
        images_512 = images_512.to(device)  # 512x512 images
        images_256 = images_256.to(device)  # 256x256 images
        images_128 = images_128.to(device)  # 128x128 images

        out_128, out_256, out_512, _ = autoencoder_mrl(images_512)

        loss_128 , loss_256 , loss_512 , loss = mrl_loss((out_128, out_256, out_512), images_128, images_256, images_512, mrl_weights)

        total_loss += loss.item()
        total_loss_128 += loss_128.item()
        total_loss_256 += loss_256.item()
        total_loss_512 += loss_512.item()


avg_loss = total_loss / len(dataloader_test_512)
avg_loss_128 = total_loss_128 / len(dataloader_test_512)
avg_loss_256 = total_loss_256 / len(dataloader_test_512)
avg_loss_512 = total_loss_512 / len(dataloader_test_512)

print(f"Total Loss: {avg_loss:.4f} | 128x128 Loss: {avg_loss_128:.4f} | 256x256 Loss: {avg_loss_256:.4f} | 512x512 Loss: {avg_loss_512:.4f}")
