# model.py
"""
Defines the neural network architecture for MNIST digit classification,
along with a utility to load pre-trained model weights.
"""

import torch
import torch.nn as nn

class MNISTBasicNetwork(nn.Module):
    """
    A simple fully connected neural network for MNIST digit classification.
    Architecture: Flatten → Linear(784→512) → ReLU → Linear(512→512) → ReLU → Linear(512→10)
    """

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
        """
        Standard forward pass through the network.
        Input: tensor of shape [batch_size, 1, 28, 28]
        Output: tensor of shape [batch_size, 10] (raw logits)
        """
        return self.layers(x)

    def predict_with_softmax(self, x):
        """
        Returns softmax probabilities instead of raw logits.
        Used for evaluation or displaying prediction confidence.
        """
        self.eval()  # Set model to eval mode to disable dropout/batchnorm
        with torch.no_grad():  # Disable gradient computation
            logits = self.forward(x)
            return nn.functional.softmax(logits, dim=1)  # Probabilities across classes

def load_trained_model(path="mnist_model.pth", device="cpu"):
    """
    Loads the trained model weights from the given file path and returns the model.
    Assumes architecture matches MNISTBasicNetwork.
    """
    model = MNISTBasicNetwork().to(device)  # Create a new model instance on correct device
    model.load_state_dict(torch.load(path, map_location=device))  # Load saved weights
    model.eval()  # Ensure model is in evaluation mode
    return model