import torch
import numpy as np
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import tqdm  

if len(sys.argv) != 5:
    print("Usage: python autoencoder.py <dataset_folder> <num_epochs> <output_folder> <save_num_epochs>")
    sys.exit(1)

dataset_folder = sys.argv[1]
num_epochs = int(sys.argv[2])
output_folder = sys.argv[3]
save_num_epochs = int(sys.argv[4])

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

data_paths = [
    os.path.join(dataset_folder, file)
    for file in os.listdir(dataset_folder)
    if file.endswith('.png') or file.endswith('.jpg')
]

dataset = []
for data_path in tqdm.tqdm(data_paths, desc='Loading dataset', unit='image'):
    image = Image.open(data_path).convert('RGB')
    image = transform(image)
    dataset.append(image)

dataset = torch.stack(dataset)

print(f"Dataset shape: {dataset.shape}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = dataset.to(device)

image_resolution = dataset[0].shape[1]

train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

print(f"Using device: {device}")

class AutoEncoder(nn.Module):
    def _init_(self):
        super(AutoEncoder, self)._init_()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 4, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(4, 4, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(4 * (image_resolution//16) * (image_resolution//16), 512),
            nn.ReLU(True),
            nn.Linear(512, 64), 
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True)           
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 512),
            nn.ReLU(True),
            nn.Linear(512, 4 * (image_resolution//16) * (image_resolution//16)),
            nn.ReLU(True),
            nn.Unflatten(1, (4, (image_resolution//16), (image_resolution//16))),
            nn.ConvTranspose2d(4, 4, 2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 8, 2, stride=2), 
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 16, 2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 2, stride=2), 
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
model = AutoEncoder().to(device) 
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num = 0

while(True):
    if os.path.exists(os.path.join(output_folder, f'autoencoder_{num}')):
        num += 1
    else:
        break

for epoch in range(num_epochs):
    model.train()
    with tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as tepoch:
        for data in tepoch:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item())
    
    if (epoch + 1) % save_num_epochs == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
        save_dir = os.path.join(output_folder, f'autoencoder_{num}')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'autoencoder_model_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at epoch {epoch+1}: {save_path}")