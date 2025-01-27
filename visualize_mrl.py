import argparse
import matplotlib.pyplot as plt
import torch
from autoencoder import AutoEncoderMRL
from torchvision import transforms
from imagedataset import ImageDataset
from torch.utils.data import DataLoader

def main(weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder_mrl = AutoEncoderMRL().to(device)
    autoencoder_mrl = torch.load(weights_path)
    autoencoder_mrl.eval()

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

    img_512 = next(iter(dataloader_512)).to(device)
    img_256 = next(iter(dataloader_256)).to(device)
    img_128 = next(iter(dataloader_128)).to(device)

    with torch.no_grad():
        out_128, out_256, out_512, _ = autoencoder_mrl(img_512)

    img_512 = img_512.cpu().numpy()
    img_256 = img_256.cpu().numpy()
    img_128 = img_128.cpu().numpy()
    out_128 = out_128.cpu().numpy()
    out_256 = out_256.cpu().numpy()
    out_512 = out_512.cpu().numpy()

    num_images = min(5, len(img_512))  

    fig, axes = plt.subplots(num_images, 6, figsize=(20, num_images * 3))

    for i in range(num_images):
        # Original 512x512
        axes[i, 0].imshow(img_512[i].transpose(1, 2, 0))  
        axes[i, 0].set_title("Original 512x512")
        axes[i, 0].axis("off")
        
        # Reconstructed 512x512
        axes[i, 1].imshow(out_512[i].transpose(1, 2, 0))
        axes[i, 1].set_title("Reconstructed 512x512")
        axes[i, 1].axis("off")
        
        # Original 256x256
        axes[i, 2].imshow(img_256[i].transpose(1, 2, 0))
        axes[i, 2].set_title("Original 256x256")
        axes[i, 2].axis("off")
        
        # Reconstructed 256x256
        axes[i, 3].imshow(out_256[i].transpose(1, 2, 0))
        axes[i, 3].set_title("Reconstructed 256x256")
        axes[i, 3].axis("off")
        
        # Original 128x128
        axes[i, 4].imshow(img_128[i].transpose(1, 2, 0))
        axes[i, 4].set_title("Original 128x128")
        axes[i, 4].axis("off")
        
        # Reconstructed 128x128
        axes[i, 5].imshow(out_128[i].transpose(1, 2, 0))
        axes[i, 5].set_title("Reconstructed 128x128")
        axes[i, 5].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MRL Autoencoder Reconstructions")
    parser.add_argument("weights_path", type=str, help="Path to the weights file")
    args = parser.parse_args()
    main(args.weights_path)