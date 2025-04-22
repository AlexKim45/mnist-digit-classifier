import torch
import torch.nn as nn

class MNISTBasicNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),                 # Flatten 28x28 image to 784-dim vector
            nn.Linear(28*28, 512),        # First hidden layer with 512 neurons
            nn.ReLU(),                    # Activation function
            nn.Linear(512, 512),          # Second hidden layer
            nn.ReLU(),                    # Another ReLU activation
            nn.Linear(512, 10)            # Output layer for 10 digit classes (0–9)
        )

    def forward(self, x):
        return self.layers(x)

    def predict_with_softmax(self, x):
        self.eval()  
        with torch.no_grad():  
            logits = self.forward(x)
            return nn.functional.softmax(logits, dim=1)  

def load_trained_model(path="mnist_model.pth", device="cpu"):
    model = MNISTBasicNetwork().to(device)  
    model.load_state_dict(torch.load(path, map_location=device)) 
    model.eval()  
    return model
