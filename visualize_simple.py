import argparse
import matplotlib.pyplot as plt
import torch
from autoencoder import AutoEncoderMRL
from torchvision import transforms
from PIL import Image
import numpy as np

def load_and_preprocess(image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    return image

def main(weights_path):
    device = torch.device("cpu")
    autoencoder_mrl = AutoEncoderMRL().to(device)
    autoencoder_mrl = torch.load(weights_path, map_location=device)
    autoencoder_mrl.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image_paths = [
        "data/train/512/-1.0000_30.0000.png",
        "data/train/512/-1.0000_153.0000.png",
        "data/train/512/-2.0000_306.0000.png",
        "data/train/512/-4.0000_215.0000.png",
        "data/train/512/-16.0000_82.0000.png",
    ]
    img_path = input("Enter the name of the image file: ")
    images = [load_and_preprocess(img_path, transform, device) for img_path in image_paths]
    images_tensor = torch.cat(images, dim=0)  # Stack into a batch

    with torch.no_grad():
        _, _, out_images, _ = autoencoder_mrl(images_tensor)

    images_np = images_tensor.cpu().numpy()
    out_images_np = out_images.cpu().numpy()

    fig, axes = plt.subplots(2, 5, figsize=(15, 6)) 

    for i in range(5):  
        axes[0, i].imshow(np.clip(images_np[i].transpose(1, 2, 0) * 0.5 + 0.5, 0, 1))  # Denormalize
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(np.clip(out_images_np[i].transpose(1, 2, 0) * 0.5 + 0.5, 0, 1))  # Denormalize
        axes[1, i].set_title(f"Reconstructed {i+1}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(img_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MRL Autoencoder Reconstructions")
    parser.add_argument("weights_path", type=str, help="Path to the weights file")
    args = parser.parse_args()
    main(args.weights_path)
