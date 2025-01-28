"""
visualize.py
Visualization module for generating heatmaps and confidence tables.
Author: GPT-4o 14.01.2025
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_heatmaps(heatmaps: List[np.ndarray], save_dir: str):
    """
    Plot and save heatmaps for activations.

    Args:
        heatmaps (List[np.ndarray]): List of heatmaps (2D arrays).
        save_dir (str): Directory to save the heatmaps.
    """
    os.makedirs(save_dir, exist_ok=True)
    for i, heatmap in enumerate(heatmaps):
        plt.figure(figsize=(6, 4))
        plt.imshow(heatmap, cmap="seismic", aspect="auto")
        plt.colorbar()
        plt.title(f"Neuron {i} Activation Heatmap")
        plt.savefig(os.path.join(save_dir, f"neuron_{i}_heatmap.png"))
        plt.close()

def plot_confidence_table(images: List[np.ndarray], labels: List[str], save_path: str):
    """
    Create a table visualization of confidence-related images.

    Args:
        images (List[np.ndarray]): List of images to plot.
        labels (List[str]): Corresponding labels for the images.
        save_path (str): Path to save the visualization.
    """
    num_labels = len(labels) // 3
    fig, axes = plt.subplots(nrows=3, ncols=num_labels, figsize=(10, 6))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(images[idx].squeeze(), cmap="gray")
        ax.set_title(labels[idx])
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Example usage
    SAVE_DIR = "../results/visualizations/"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Dummy data for demonstration purposes
    dummy_heatmaps = [np.random.rand(10, 10) for _ in range(10)]
    dummy_images = [np.random.rand(28, 28) for _ in range(30)]
    dummy_labels = [f"Label {i//3} - {['Min', 'Max', 'Avg'][i%3]}" for i in range(30)]

    # Plot and save heatmaps
    plot_heatmaps(dummy_heatmaps, SAVE_DIR)

    # Plot and save confidence table
    plot_confidence_table(dummy_images, dummy_labels, os.path.join(SAVE_DIR, "confidence_table.png"))
