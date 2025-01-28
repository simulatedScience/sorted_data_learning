"""
train.py
Training module for baseline and sorted training experiments on MNIST.
Author: GPT-4o 14.01.2025
"""

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import List

def load_data(data_dir: str, batch_size: int, shuffle: bool = False) -> DataLoader:
    """
    Load the MNIST dataset from the specified directory.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Number of samples per batch.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transforms.ToTensor()
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class SimpleNN(nn.Module):
    """
    A simple neural network with no hidden layers for MNIST classification.
    """
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

def train_model(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    epochs: int, 
    save_path: str
):
    """
    Train the given model.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training dataset.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        epochs (int): Number of training epochs.
        save_path (str): Path to save the trained model.
    """
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss:.4f}")

    # Save the trained model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # Paths and configurations
    DATA_DIR = "../data/sorted/ascending/"  # Example path
    MODEL_SAVE_PATH = "../results/models/simple_nn.pth"
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.001

    # Load data
    dataloader = load_data(DATA_DIR, BATCH_SIZE)

    # Initialize model, loss function, and optimizer
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, EPOCHS, MODEL_SAVE_PATH)