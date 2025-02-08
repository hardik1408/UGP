from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from autoencoder import Encoder, Decoder, AutoEncoderMRL
from imagedataset import ImageDataset
import torch
import torch.nn.functional as F
import time

# ---------------------- SETTINGS ----------------------
batch_size = 32
num_epochs = int(input("Enter number of epochs: "))
# num_epochs = 10
mrl_weights = [1, 1, 1]
device = torch.device("cpu")
print(f"Using device: {device}")

# ---------------------- TRANSFORMS ----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# ---------------------- FUNCTION TO PRELOAD DATASET ----------------------
def preload_dataset(folder_path, transform, device):
    dataset = ImageDataset(folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preloaded_data = []
    for batch in dataloader:
        preloaded_data.append(batch.to(device))  # Move images to GPU
    return preloaded_data

# Load all images before training
print("Preloading datasets to GPU (this may take time)...")
preloaded_512 = preload_dataset('data/train_new/512', transform, device)
preloaded_256 = preload_dataset('data/train_new/256', transform, device)
preloaded_128 = preload_dataset('data/train_new/128', transform, device)
print("Preloading complete!")

# ---------------------- MULTI-RESOLUTION LOSS FUNCTION ----------------------
def mrl_loss(outputs, target_128, target_256, target_512, weights):
    out_128, out_256, out_512 = outputs
    loss_128 = F.mse_loss(out_128, target_128)
    loss_256 = F.mse_loss(out_256, target_256)
    loss_512 = F.mse_loss(out_512, target_512)

    total_loss = weights[0] * loss_128 + weights[1] * loss_256 + weights[2] * loss_512
    return loss_128, loss_256, loss_512, total_loss / 3

# ---------------------- TRAINING ----------------------
autoencoder_mrl = AutoEncoderMRL().to(device)
optimizer = torch.optim.Adam(autoencoder_mrl.parameters(), lr=1e-3)

print("Starting training...")
for epoch in range(num_epochs):
    autoencoder_mrl.train()
    total_loss = 0
    total_loss_128 = 0
    total_loss_256 = 0
    total_loss_512 = 0
    start_time = time.time()

    for batch_idx, (images_512, images_256, images_128) in enumerate(zip(preloaded_512, preloaded_256, preloaded_128)):
        optimizer.zero_grad()

        out_128, out_256, out_512, _ = autoencoder_mrl(images_512)
        loss_128, loss_256, loss_512, loss = mrl_loss(
            (out_128, out_256, out_512),
            images_128, images_256, images_512,
            mrl_weights
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_loss_128 += loss_128.item()
        total_loss_256 += loss_256.item()
        total_loss_512 += loss_512.item()

    end_time = time.time()
    avg_loss = total_loss / len(preloaded_512)
    avg_loss_128 = total_loss_128 / len(preloaded_512)
    avg_loss_256 = total_loss_256 / len(preloaded_512)
    avg_loss_512 = total_loss_512 / len(preloaded_512)

    print(f"Epoch [{epoch+1}/{num_epochs}]: Total Loss: {avg_loss:.4f} | "
          f"128x128 Loss: {avg_loss_128:.4f} | 256x256 Loss: {avg_loss_256:.4f} | 512x512 Loss: {avg_loss_512:.4f} | "
          f"Time: {end_time - start_time:.2f}s")

torch.save(autoencoder_mrl, 'gpu_ae.pth')
print("Training complete!")
