from torchmetrics.functional import peak_signal_noise_ratio as psnr
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from autoencoder import Encoder, Decoder, AutoEncoderMRL
from imagedataset import ImageDataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys
import torch.nn.functional as F

if len(sys.argv) != 3:
    print("Usage: python3 test.py <device> <weights_path>")
    sys.exit(1)

device_name = sys.argv[1]
weights_path = sys.argv[2]

device = torch.device(device_name)
def mrl_loss(outputs, target_128, target_256, target_512, weights):
    out_128, out_256, out_512 = outputs
    loss_128 = F.mse_loss(out_128, target_128)
    loss_256 = F.mse_loss(out_256, target_256)
    loss_512 = F.mse_loss(out_512, target_512)

    # print(f"PSNR Loss at 128x128: {-loss_128:.4f}, 256x256: {-loss_256:.4f}, 512x512: {-loss_512:.4f}")

    total_loss = weights[0] * loss_128 + weights[1] * loss_256 + weights[2] * loss_512
    return loss_128 , loss_256 , loss_512 , total_loss / 3 


autoencoder_mrl = AutoEncoderMRL().to(device)
autoencoder_mrl = torch.load(weights_path)
autoencoder_mrl.eval()
total_loss = 0

transform = transforms.Compose([
    transforms.ToTensor(),          # Convert image to PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

dataset_test_512 = ImageDataset("data/test_new/512", transform=transform)
dataset_test_256 = ImageDataset("data/test_new/256", transform=transform) 
dataset_test_128 = ImageDataset("data/test_new/128", transform=transform) 
batch_size = 32
dataloader_test_512 = DataLoader(dataset_test_512, batch_size=batch_size, shuffle=False)
dataloader_test_256 = DataLoader(dataset_test_256, batch_size=batch_size, shuffle=False)
dataloader_test_128 = DataLoader(dataset_test_128, batch_size=batch_size, shuffle=False)

mrl_weights = [1,1,1]
total_loss = 0
total_loss_128 = 0
total_loss_256 = 0
total_loss_512 = 0
total_psnr_128 = 0
total_psnr_256 = 0
total_psnr_512 = 0


def psnr_loss(output, target):
    mse = F.mse_loss(output, target, reduction='mean')
    psnr = -10 * torch.log10(mse + 1e-8)  # Adding small epsilon to avoid log(0)
    return psnr 

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

        total_psnr_128 += psnr_loss(out_128, images_128).item()
        total_psnr_256 += psnr_loss(out_256, images_256).item()
        total_psnr_512 += psnr_loss(out_512, images_512).item()


avg_loss = total_loss / len(dataloader_test_512)
avg_loss_128 = total_loss_128 / len(dataloader_test_512)
avg_loss_256 = total_loss_256 / len(dataloader_test_512)
avg_loss_512 = total_loss_512 / len(dataloader_test_512)
avg_psnr_128 = total_psnr_128 / len(dataloader_test_512)
avg_psnr_256 = total_psnr_256 / len(dataloader_test_512)
avg_psnr_512 = total_psnr_512 / len(dataloader_test_512)

print(f"Total Loss: {avg_loss:.4f} | 128x128 Loss: {avg_loss_128:.4f} | 256x256 Loss: {avg_loss_256:.4f} | 512x512 Loss: {avg_loss_512:.4f}")
print(f"Average PSNR: 128x128: {avg_psnr_128:.4f} | 256x256: {avg_psnr_256:.4f} | 512x512: {avg_psnr_512:.4f}")