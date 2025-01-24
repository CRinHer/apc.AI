print("os importing...")
import os
print("PIL importing...")
from PIL import Image
print("sklearn importing...")
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.animation as animation

print("torch importing...")
import torch 
import torchvision 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

print("imports done")

# Define the image directory
image_dir = '../generate/preprocessing/Images'  # Replace with the actual path

# Define the dataset
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, filename) 
                            for filename in os.listdir(image_dir) if filename.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image = image.resize((200, 200))  # Resize if needed
        image = transforms.ToTensor()(image)  # Convert to PyTorch tensor
        return image

# Create the dataset
print("dataset generating...")
dataset = ImageDataset(image_dir)
print("dataset generated")

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=200, shuffle=True)

# Split the dataset into training and testing sets
train_size = 40000
test_size = 8000

train_indices = list(range(train_size))
test_indices = list(range(train_size, train_size + test_size))

train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# Create DataLoaders for training and testing
train_dataloader = DataLoader(train_dataset, batch_size=200, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=200, shuffle=False) 

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(20, 20), padding=9, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((40, 40)),
            nn.Conv2d(40, 20, kernel_size=(2, 2), padding=0, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((20, 20)),
            nn.Conv2d(20, 10, kernel_size=(5, 5), padding=2, stride=1),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(10, 20, kernel_size=(5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(20, 20, kernel_size=(2, 2), stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=5),
            nn.ConvTranspose2d(20, 1, kernel_size=(20, 20), stride=1, padding=7),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# # Create the autoencoder model
# autoencoder = Autoencoder()

# # Compile the model and define the optimizer
# optimizer = optim.Adam(autoencoder.parameters())
# loss_fn = nn.BCELoss()  # Binary Cross Entropy loss for grayscale images

# Train the model
def print_loss(epoch, loss):
    print(f"Epoch: {epoch+1}, Loss: {loss}")

# for epoch in range(5):
#     round = 1
#     print(f"Epoch {epoch}")
#     total_loss = 0.0
#     for data in train_dataloader:
#         # Forward pass
#         outputs = autoencoder(data)
#         loss = loss_fn(outputs, data)
#         print(f"Round {round}: {loss}")

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#         round += 1

#     # Print average loss for the epoch
#     print_loss(epoch, total_loss / len(train_dataloader))

# You can add code here to save the trained model or use it for inference
# total_test_loss = 0.0
# for data in test_dataloader:
#     outputs = autoencoder(data)
#     loss = loss_fn(outputs, data)
#     total_test_loss += loss.item()
# print(f"Test Loss: {total_test_loss / len(test_dataloader)}")

# torch.save(autoencoder.state_dict(), 'autoencoder_model.pth')

# Load the saved state dictionary
state_dict = torch.load('autoencoder_model.pth')
# Create a new instance of the encoder model
encoder = Autoencoder()
# Load the state dictionary into the encoder
encoder.load_state_dict(state_dict)
encoder = encoder.encoder


# encoder = autoencoder.encoder  # Get the encoder part

print("Model Loaded")
print("Dataset encoding...")
# Get encoded representations for your dataset
encoded_data = []
for data in test_dataloader:  # Or iterate over your test_dataloader
    with torch.no_grad():
        encoded_batch = encoder(data)
        encoded_data.append(encoded_batch)

print("Dataset Encoded")

# Concatenate the encoded batches into a single tensor
encoded_data = torch.cat(encoded_data, dim=0)
encoded_data = encoded_data.reshape(-1, encoded_data.shape[1] * encoded_data.shape[2] * encoded_data.shape[3])

print("Clustering...")
# Assuming you want to find 3 clusters
kmeans = KMeans(n_clusters=2)
cluster_labels = kmeans.fit_predict(encoded_data.numpy()) 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the clusters in 3D
ax.scatter(encoded_data[:, 0], encoded_data[:, 1], encoded_data[:, 2], c=cluster_labels)  # Assuming 3 features

ax.set_title('KMeans Clustering (3D)')
ax.set_xlabel('Oscillation') # How wildly does it oscillate?
ax.set_ylabel('Conformance') # How quickly does it reach steady state?
ax.set_zlabel('Stability') # Does it reach steady state?

def animate(i):
    ax.view_init(elev=30, azim=i)
    return fig,

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=360, interval=20, blit=True)

plt.show() 