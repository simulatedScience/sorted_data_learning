"""
analysis.py
Analysis tools for evaluating neural network performance and interpretability.
Author: GPT-4o 14.01.2025
"""

import os
from typing import Tuple, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from train import SimpleNN

def visualize_activations(model: nn.Module, dataloader: DataLoader, save_dir: str):
    """
    Visualize neuron activations for single-layer neural networks.

    Args:
        model (nn.Module): Trained neural network model.
        dataloader (DataLoader): DataLoader for the dataset.
        save_dir (str): Directory to save the heatmaps.
    """
    model.eval()
    activations = []
    with torch.no_grad():
        for images, _ in dataloader:
            outputs = model(images)
            activations.append(outputs)

    activations = torch.cat(activations, dim=0).numpy()
    os.makedirs(save_dir, exist_ok=True)

    for i in range(10):
        plt.figure(figsize=(6, 4))
        plt.imshow(activations[:, i].reshape(-1, 1), cmap='seismic', aspect='auto')
        plt.colorbar()
        plt.title(f"Neuron {i} Activations")
        plt.savefig(os.path.join(save_dir, f"neuron_{i}_activation.png"))
        plt.close()

def confidence_analysis(
    model: nn.Module, dataloader: DataLoader, save_dir: str
):
    """
    Perform confidence analysis for multi-layer neural networks.

    Args:
        model (nn.Module): Trained neural network model.
        dataloader (DataLoader): DataLoader for the dataset.
        save_dir (str): Directory to save the confidence images.
    """
    model.eval()
    confidence_data = {label: [] for label in range(10)}
    avg_images = {label: [] for label in range(10)}

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            confidences, predictions = torch.max(outputs, dim=1)
            second_confidences = torch.topk(outputs, 2, dim=1).values[:, 1]
            
            for i in range(len(images)):
                conf_diff = confidences[i] - second_confidences[i]
                label = labels[i].item()
                confidence_data[label].append((images[i], conf_diff))

    os.makedirs(save_dir, exist_ok=True)
    
    for label in range(10):
        confidence_data[label].sort(key=lambda x: x[1])  # Sort by confidence difference
        min_conf_img = confidence_data[label][0][0]
        max_conf_img = confidence_data[label][-1][0]
        avg_image = torch.mean(torch.stack([x[0] for x in confidence_data[label]]), dim=0)

        # Save images
        for name, img_tensor in zip(["min", "max", "avg"], [min_conf_img, max_conf_img, avg_image]):
            img_path = os.path.join(save_dir, f"label_{label}_{name}.png")
            plt.imshow(img_tensor.squeeze().numpy(), cmap="gray")
            plt.title(f"Label {label} - {name} confidence")
            plt.savefig(img_path)
            plt.close()

if __name__ == "__main__":
    # Paths and configurations
    DATA_DIR = "../data/sorted/ascending/"  # Example path
    MODEL_PATH = "../results/models/simple_nn.pth"
    SAVE_DIR_ACTIVATIONS = "../results/activations/"
    SAVE_DIR_CONFIDENCE = "../results/confidence_images/"
    BATCH_SIZE = 64

    # Load data
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = SimpleNN()
    model.load_state_dict(torch.load(MODEL_PATH))

    # Perform analyses
    visualize_activations(model, test_loader, SAVE_DIR_ACTIVATIONS)
    confidence_analysis(model, test_loader, SAVE_DIR_CONFIDENCE)
