import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import sys
from autoencoder import AutoEncoderMRL

if len(sys.argv) != 4:
    print("Usage: python3 image.py <device> <weights_path> <image_path>")
    sys.exit(1)

device_name = sys.argv[1]
weights_path = sys.argv[2]
image_path = sys.argv[3]

def test_autoencoder_on_image(image_path, autoencoder, device):
    image = Image.open(image_path).convert('RGB')  
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    input_image = preprocess(image).unsqueeze(0).to(device)  
    
    autoencoder.eval() 
    with torch.no_grad():
        out_128, out_256, reconstructed_image, _ = autoencoder(input_image)
    
    input_image = input_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    reconstructed_image = reconstructed_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    # input_image = (input_image * 0.5) + 0.5
    # reconstructed_image = (reconstructed_image * 0.5) + 0.5


    plt.figure(figsize=(8, 4))
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(input_image)
    plt.title("Original Image")
    plt.axis('off')
    # Reconstructed image
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image)
    plt.title("Reconstructed Image")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

from autoencoder import AutoEncoderMRL
device = torch.device(device_name)
autoencoder_mrl = AutoEncoderMRL().to(device)
autoencoder_mrl = torch.load(weights_path)
autoencoder_mrl.eval()
test_autoencoder_on_image(image_path, autoencoder_mrl, device)