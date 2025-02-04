import argparse
import matplotlib.pyplot as plt
import torch
from autoencoder import AutoEncoderMRL
from torchvision import transforms
from imagedataset import ImageDataset
from torch.utils.data import DataLoader

def main(weights_path):
    device = torch.device('cpu')
    autoencoder_mrl = AutoEncoderMRL().to(device)
    autoencoder_mrl = torch.load(weights_path)
    autoencoder_mrl.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),          # Convert image to PyTorch tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])


    folder_path_512 = 'data/train/512'

    dataset_512 = ImageDataset(folder_path_512, transform=transform)

    batch_size = 32
    dataloader_512 = DataLoader(dataset_512, batch_size=batch_size, shuffle=False)

    img_512 = next(iter(dataloader_512)).to(device)
    with torch.no_grad():
        _, _, out_512, _ = autoencoder_mrl(img_512)

    img_512 = img_512.cpu().numpy()
    out_512 = out_512.cpu().numpy()

    num_images = min(5, len(img_512))  

    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 3))

    for i in range(num_images):
        # Original 512x512
        axes[i, 0].imshow(img_512[i].transpose(1, 2, 0))  
        axes[i, 0].set_title("Original 512x512")
        axes[i, 0].axis("off")
        
        # Reconstructed 512x512
        axes[i, 1].imshow(out_512[i].transpose(1, 2, 0))
        axes[i, 1].set_title("Reconstructed 512x512")
        axes[i, 1].axis("off")
        

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MRL Autoencoder Reconstructions")
    parser.add_argument("weights_path", type=str, help="Path to the weights file")
    args = parser.parse_args()
    main(args.weights_path)