from torchmetrics.functional import peak_signal_noise_ratio as psnr
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
import sys
import torch.nn.functional as F

if len(sys.argv) != 3:
    print("Usage: python3 test_simpe.py <device> <weights_path>")
    sys.exit(1)

device_name = sys.argv[1]
weights_path = sys.argv[2]

device = torch.device(device_name)
transform = transforms.Compose([
    transforms.ToTensor(),          # Convert image to PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

dataset_test_512 = ImageDataset("data/test_new/512", transform=transform)

batch_size = 32
dataloader_test_512 = DataLoader(dataset_test_512, batch_size=batch_size, shuffle=False)

total_loss = 0

autoencoder_simple = AutoEncoderMRL().to(device)
autoencoder_simple = torch.load(weights_path)
autoencoder_simple.eval()
total_loss = 0
psnr_val = 0

def psnr_loss(output, target):
    mse = F.mse_loss(output, target, reduction='mean')
    psnr = -10 * torch.log10(mse + 1e-8)  # Adding small epsilon to avoid log(0)
    return psnr 



with torch.no_grad():
    for images_512 in dataloader_test_512:
        images_512 = images_512.to(device)  # 512x512 images

        _, _, out_512, _ = autoencoder_simple(images_512)
        loss = F.mse_loss(out_512, images_512)
        psnr_val += psnr_loss(out_512, images_512)


        total_loss += loss.item()

avg_loss = total_loss / len(dataloader_test_512)
avg_psnr = psnr_val / len(dataloader_test_512)  
print(f"Total Loss: {avg_loss:.4f}")
print(f"Average PSNR: {avg_psnr:.4f}")
