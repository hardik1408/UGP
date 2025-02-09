import matplotlib.pyplot as plt
import re

# Path to the uploaded log file
log_file_path = "terminal/week4/output_downsample_300.log"

# Lists to store parsed data
epochs = []
loss_128 = []
loss_256 = []
loss_512 = []

# Regular expression to match the loss values
epoch_pattern = re.compile(r"Epoch \[(\d+)/\d+\]: .*128x128 Loss: ([\d\.]+) \| 256x256 Loss: ([\d\.]+) \| 512x512 Loss: ([\d\.]+)")

# Read and parse the log file
with open(log_file_path, "r") as file:
    for line in file:
        match = epoch_pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            loss_128.append(float(match.group(2)))
            loss_256.append(float(match.group(3)))
            loss_512.append(float(match.group(4)))

# Lists to store total loss values
total_loss = [(l1 + l2 + l3)/3 for l1, l2, l3 in zip(loss_128, loss_256, loss_512)]

# Plot the losses with bold lines
plt.figure(figsize=(10, 5))
plt.plot(epochs, loss_128, label="128x128 Loss", color='r', linewidth=2)
plt.plot(epochs, loss_256, label="256x256 Loss", color='g', linewidth=2)
plt.plot(epochs, loss_512, label="512x512 Loss", color='b', linewidth=2)
plt.plot(epochs, total_loss, label="Total Loss", color='k', linewidth=3, linestyle='dashed')

plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Loss vs Epoch for Different Resolutions", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

