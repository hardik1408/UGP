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
# def mrl_loss(outputs, target_128, target_256, target_512, weights):
#     out_128, out_256, out_512 = outputs
#     loss_128 = -psnr(out_128, target_128, data_range=1.0)
#     loss_256 = -psnr(out_256, target_256, data_range=1.0)
#     loss_512 = -psnr(out_512, target_512, data_range=1.0)

#     # print(f"PSNR Loss at 128x128: {-loss_128:.4f}, 256x256: {-loss_256:.4f}, 512x512: {-loss_512:.4f}")

#     # Weighted sum of losses
#     total_loss = weights[0] * loss_128 + weights[1] * loss_256 + weights[2] * loss_512
#     return loss_128 , loss_256 , loss_512 , total_loss / 3  

transform = transforms.Compose([
    transforms.ToTensor(),          # Convert image to PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

folder_path_512 = 'data/train_new/512'
folder_path_256 = 'data/train_new/256'
folder_path_128 = 'data/train_new/128'
dataset_512 = ImageDataset(folder_path_512, transform=transform)
dataset_256 = ImageDataset(folder_path_256, transform=transform)
dataset_128 = ImageDataset(folder_path_128, transform=transform)


batch_size = 32
dataloader_512 = DataLoader(dataset_512, batch_size=batch_size, shuffle=False)
dataloader_256 = DataLoader(dataset_256, batch_size=batch_size, shuffle=False)
dataloader_128 = DataLoader(dataset_128, batch_size=batch_size, shuffle=False)

def visualize_images(dataloader, title, num_images=8):

    images = next(iter(dataloader))[:num_images]
    grid = vutils.make_grid(images, nrow=num_images, normalize=True, scale_each=True)
    # print(images.shape)
    grid_np = grid.cpu().numpy().transpose(1, 2, 0)  

    plt.figure(figsize=(15, 5))
    plt.imshow(grid_np)
    plt.title(title)
    plt.axis('off')
    plt.show()

# visualize_images(dataloader_512, "Sample Images (512x512)")
# visualize_images(dataloader_256, "Sample Images (256x256)")
# visualize_images(dataloader_128, "Sample Images (128x128)")

device = torch.device('cpu')
autoencoder_simple = AutoEncoderMRL().to(device)

optimizer = Adam(autoencoder_simple.parameters(), lr=1e-3)

num_epochs = 20
for epoch in range(num_epochs):
    autoencoder_simple.train()  
    total_loss = 0
    s_time = time.time()
    for images_512 in dataloader_512:
        images_512 = images_512.to(device)

        _, _, out_512, _ = autoencoder_simple(images_512)

        loss = F.mse_loss(out_512, images_512)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader_512)
    e_time = time.time()
    print(f"time for {epoch+1} epoch = {e_time - s_time}")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    if((epoch+1)%10 == 0):
        torch.save(autoencoder_simple,'weights/simple_big_ae.pth')
        print(f"model for {epoch+1} saved")

torch.save(autoencoder_simple,'simple_big_ae.pth')
print("training complete")