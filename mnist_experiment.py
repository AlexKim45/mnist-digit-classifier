import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
import time

try:
    import google.colab
    running_on_colab = True
except ImportError:
    running_on_colab = False

device = "cuda" if torch.cuda.is_available() else "cpu"


MEAN = 0.1307
STD_DEV = 0.3081
NUM_EPOCHS = 8
LEARNING_RATE = 0.02
BATCH_SIZE = 128
PRINT_EVERY = 1  

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD_DEV)
])

train_dataset = datasets.MNIST("MNIST_data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("MNIST_data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class MNISTBasicNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.layers(x)

    def predict_with_softmax(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return nn.functional.softmax(logits, dim=1)

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

        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if epoch % PRINT_EVERY == 0:
        avg_loss = total_loss / total_samples
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")

model.eval()
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()

accuracy = 100 * correct / len(test_loader.dataset)
print(f"Test Accuracy: {accuracy:.2f}%")

rand_idx = random.randint(0, len(test_dataset) - 1)
image, true_label = test_dataset[rand_idx]
image = image.to(device)

probs = model.predict_with_softmax(image.unsqueeze(0)) 
predicted_label = torch.argmax(probs, dim=1).item()

print(f"\nActual Label: {true_label}")
print(f"Model's Guess: {predicted_label}\n")

for digit, prob in enumerate(probs.squeeze().cpu().tolist()):
    print(f"{digit}: {prob * 100:.4f}%")

unnormalized_img = image * STD_DEV + MEAN
plt.imshow(unnormalized_img.squeeze(0).cpu(), cmap="gray")
plt.title(f"Predicted: {predicted_label} | Actual: {true_label}")
plt.axis("off")
plt.show()
