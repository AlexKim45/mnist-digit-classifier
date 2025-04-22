# train.py
"""
Train a simple feedforward neural network to classify MNIST digits.
This script loads the MNIST dataset, trains the model over multiple epochs,
evaluates its performance, and saves the trained weights.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

from model import MNISTBasicNetwork  # Import the model architecture

# Set device to GPU if available, else fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants (hyperparameters and normalization settings)
MEAN = 0.1307           # Mean of MNIST dataset
STD_DEV = 0.3081        # Std dev of MNIST dataset
NUM_EPOCHS = 8          # Total training passes over the dataset
LEARNING_RATE = 0.02    # Learning rate for gradient descent
BATCH_SIZE = 128        # Number of samples per training batch

# Define preprocessing transform: convert to tensor and normalize pixel values
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD_DEV)
])

# Load training and testing datasets
train_dataset = datasets.MNIST("MNIST_data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("MNIST_data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Instantiate model and move it to the correct device
model = MNISTBasicNetwork().to(device)

# Define loss function (CrossEntropy is standard for multi-class classification)
loss_fn = nn.CrossEntropyLoss()

# Use SGD optimizer to update model parameters
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Start training
start_time = time.time()
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()  # Set model to training mode
    total_loss = 0
    total_samples = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move batch to device

        optimizer.zero_grad()          # Reset gradients from previous step
        outputs = model(images)        # Forward pass
        loss = loss_fn(outputs, labels)  # Compute loss
        loss.backward()               # Backward pass (compute gradients)
        optimizer.step()              # Apply gradient step to update model

        total_loss += loss.item() * images.size(0)  # Accumulate weighted loss
        total_samples += images.size(0)             # Track total samples seen

    # Report average loss for this epoch
    avg_loss = total_loss / total_samples
    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

print(f"Training completed in {time.time() - start_time:.2f} seconds")

# Evaluate the model on the test set
model.eval()  # Switch to evaluation mode
correct = 0
with torch.no_grad():  # Disable gradient tracking for evaluation
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        predictions = torch.argmax(model(images), dim=1)  # Pick class with highest score
        correct += (predictions == labels).sum().item()   # Count correct predictions

# Compute and print final accuracy
accuracy = 100 * correct / len(test_loader.dataset)
print(f"Test Accuracy: {accuracy:.2f}%")

# Save trained model weights to file
torch.save(model.state_dict(), "mnist_model.pth")
print("Model saved as mnist_model.pth")