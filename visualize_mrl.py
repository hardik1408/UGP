import argparse
import matplotlib.pyplot as plt
import torch
from autoencoder import AutoEncoderMRL
from torchvision import transforms
from PIL import Image
import numpy as np

def load_and_preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")  # Ensure image is RGB
    return transform(image).unsqueeze(0)  # Add batch dimension

def main(weights_path):
    device = torch.device('cpu')
    autoencoder_mrl = AutoEncoderMRL().to(device)
    autoencoder_mrl = torch.load(weights_path, map_location=torch.device('cpu'))
    autoencoder_mrl.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])

    image_paths_512 = [
        "data/train/512/-1.0000_30.0000.png",
        "data/train/512/-1.0000_153.0000.png",
        "data/train/512/-2.0000_306.0000.png",
        "data/train/512/-4.0000_215.0000.png",
        "data/train/512/-16.0000_82.0000.png",
    ]

    image_paths_256 = [
        "data/train/256/-1.0000_30.0000.png",
        "data/train/256/-1.0000_153.0000.png",
        "data/train/256/-2.0000_306.0000.png",
        "data/train/256/-4.0000_215.0000.png",
        "data/train/256/-16.0000_82.0000.png",
    ]

    image_paths_128 = [
        "data/train/128/-1.0000_30.0000.png",
        "data/train/128/-1.0000_153.0000.png",
        "data/train/128/-2.0000_306.0000.png",
        "data/train/128/-4.0000_215.0000.png",
        "data/train/128/-16.0000_82.0000.png",
    ]

    img_512 = torch.cat([load_and_preprocess_image(p, transform) for p in image_paths_512]).to(device)
    img_256 = torch.cat([load_and_preprocess_image(p, transform) for p in image_paths_256]).to(device)
    img_128 = torch.cat([load_and_preprocess_image(p, transform) for p in image_paths_128]).to(device)
    img_name = input("Enter the name of the image file: ")
    with torch.no_grad():
        out_128, out_256, out_512, _ = autoencoder_mrl(img_512)

    img_512 = img_512.cpu().numpy()
    img_256 = img_256.cpu().numpy()
    img_128 = img_128.cpu().numpy()
    out_128 = out_128.cpu().numpy()
    out_256 = out_256.cpu().numpy()
    out_512 = out_512.cpu().numpy()

    fig, axes = plt.subplots(5, 6, figsize=(20, 15))  # 5 images, 6 columns

    for i in range(5):
        axes[i, 0].imshow(img_512[i].transpose(1, 2, 0))  
        axes[i, 0].set_title("Original 512x512")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(out_512[i].transpose(1, 2, 0))
        axes[i, 1].set_title("Reconstructed 512x512")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(img_256[i].transpose(1, 2, 0))
        axes[i, 2].set_title("Original 256x256")
        axes[i, 2].axis("off")
        
        axes[i, 3].imshow(out_256[i].transpose(1, 2, 0))
        axes[i, 3].set_title("Reconstructed 256x256")
        axes[i, 3].axis("off")
        
        axes[i, 4].imshow(img_128[i].transpose(1, 2, 0))
        axes[i, 4].set_title("Original 128x128")
        axes[i, 4].axis("off")
        
        axes[i, 5].imshow(out_128[i].transpose(1, 2, 0))
        axes[i, 5].set_title("Reconstructed 128x128")
        axes[i, 5].axis("off")

    plt.tight_layout()
    plt.savefig(img_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MRL Autoencoder Reconstructions")
    parser.add_argument("weights_path", type=str, help="Path to the weights file")
    args = parser.parse_args()
    main(args.weights_path)
