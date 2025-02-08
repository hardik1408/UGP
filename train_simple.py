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
 

transform = transforms.Compose([
    transforms.ToTensor(),          # Convert image to PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

folder_path_512 = 'data/train/512'
dataset_512 = ImageDataset(folder_path_512, transform=transform)

batch_size = 32
dataloader_512 = DataLoader(dataset_512, batch_size=batch_size, shuffle=False)

device = torch.device('cuda')
autoencoder_simple = AutoEncoderMRL().to(device)

optimizer = Adam(autoencoder_simple.parameters(), lr=1e-3)

num_epochs = 80
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
    # if((epoch+1)%10 == 0):
    #     torch.save(autoencoder_simple,'weights/simple_big_ae.pth')
    #     print(f"model for {epoch+1} saved")

torch.save(autoencoder_simple,'weights/gpu_simple_80.pth')
print("training complete")