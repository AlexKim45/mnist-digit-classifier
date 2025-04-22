import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

from model import MNISTBasicNetwork  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
MEAN = 0.1307           # Mean of MNIST dataset
STD_DEV = 0.3081        # Std dev of MNIST dataset
NUM_EPOCHS = 8          
LEARNING_RATE = 0.02    
BATCH_SIZE = 128        

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD_DEV)
])

train_dataset = datasets.MNIST("MNIST_data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("MNIST_data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = MNISTBasicNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

start_time = time.time()
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()  
    total_loss = 0
    total_samples = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  

        optimizer.zero_grad()         
        outputs = model(images)        
        loss = loss_fn(outputs, labels)  
        loss.backward()               
        optimizer.step()              

        total_loss += loss.item() * images.size(0) 
        total_samples += images.size(0)           

    # Report average loss for this epoch
    avg_loss = total_loss / total_samples
    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

print(f"Training completed in {time.time() - start_time:.2f} seconds")

# Evaluate the model on the test set
model.eval()  
correct = 0
with torch.no_grad():  
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        predictions = torch.argmax(model(images), dim=1)  # Pick class with highest score
        correct += (predictions == labels).sum().item()   # Count correct predictions

accuracy = 100 * correct / len(test_loader.dataset)
print(f"Test Accuracy: {accuracy:.2f}%")

# Save trained model weights to file
torch.save(model.state_dict(), "mnist_model.pth")
print("Model saved as mnist_model.pth")
