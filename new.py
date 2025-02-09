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
def mrl_loss(outputs, target_128, target_256, target_512, weights):
    out_128, out_256, out_512 = outputs
    loss_128 = F.mse_loss(out_128, target_128)
    loss_256 = F.mse_loss(out_256, target_256)
    loss_512 = F.mse_loss(out_512, target_512)

    # print(f"PSNR Loss at 128x128: {-loss_128:.4f}, 256x256: {-loss_256:.4f}, 512x512: {-loss_512:.4f}")

    # Weighted sum of losses
    total_loss = weights[0] * loss_128 + weights[1] * loss_256 + weights[2] * loss_512
    return loss_128 , loss_256 , loss_512 , total_loss / 3  

transform = transforms.Compose([
    transforms.ToTensor(),          # Convert image to PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

folder_path_512 = 'data/train/512'
folder_path_256 = 'data/train/256'
folder_path_128 = 'data/train/128'

dataset_512 = ImageDataset(folder_path_512, transform=transform)
dataset_256 = ImageDataset(folder_path_256, transform=transform)
dataset_128 = ImageDataset(folder_path_128, transform=transform)

batch_size = 32
dataloader_512 = DataLoader(dataset_512, batch_size=batch_size, shuffle=False)
dataloader_256 = DataLoader(dataset_256, batch_size=batch_size, shuffle=False)
dataloader_128 = DataLoader(dataset_128, batch_size=batch_size, shuffle=False)


torch.cuda.empty_cache()
num_epochs = 300
mrl_weights = [1, 1, 1] 

device = torch.device('cuda')
print(f"device is {device} ")
autoencoder_mrl = AutoEncoderMRL().to(device)
#print(autoencoder_mrl)
print("going into training")
optimizer = Adam(autoencoder_mrl.parameters(), lr=1e-3)
for epoch in range(num_epochs):
    autoencoder_mrl.train()  
    total_loss = 0
    total_loss_128 = 0
    total_loss_256 = 0
    total_loss_512 = 0
    start_time = time.time()
    for batch_idx, (images_512, images_256, images_128) in enumerate(zip(dataloader_512, dataloader_256, dataloader_128)):
        images_512 = images_512.to(device)  #
        images_256 = images_256.to(device)  
        images_128 = images_128.to(device)  

        out_128, out_256, out_512, _ = autoencoder_mrl(images_512)
        # print(f"out_128 shape: {out_128.shape}")
        # print(f"out_256 shape: {out_256.shape}")
        # print(f"out_512 shape: {out_512.shape}")

        loss_128 , loss_256 , loss_512 , loss = mrl_loss((out_128, out_256, out_512), images_128, images_256, images_512, mrl_weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_loss_128 += loss_128.item()
        total_loss_256 += loss_256.item()
        total_loss_512 += loss_512.item()

    end_time = time.time()
    avg_loss = total_loss / len(dataloader_512)
    avg_loss_128 = total_loss_128 / len(dataloader_512)
    avg_loss_256 = total_loss_256 / len(dataloader_512)
    avg_loss_512 = total_loss_512 / len(dataloader_512)
    print(f"epoch time : {end_time - start_time}")
    print(f"Epoch [{epoch+1}/{num_epochs}]: Total Loss: {avg_loss:.4f} | 128x128 Loss: {avg_loss_128:.4f} | 256x256 Loss: {avg_loss_256:.4f} | 512x512 Loss: {avg_loss_512:.4f}")
    # if((epoch+1)%10 == 0):
    #     torch.save(autoencoder_mrl,'weights/big_ae.pth')
    #     print(f"model for {epoch+1} saved")

torch.save(autoencoder_mrl,'weights/gpu_300.pth')
print("training complete")
