"""
preprocess.py
Preprocessing module for sorting and shuffling MNIST training data.
Author: GPT-4o 14.01.2025
"""

import os
import random
from typing import List
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset

def sort_mnist_data(dataset: Dataset, order: List[int]) -> List[np.ndarray]:
    """
    Sort the MNIST dataset based on the given label order.

    Args:
        dataset (Dataset): The MNIST dataset to sort.
        order (List[int]): The label order to sort by.

    Returns:
        List[np.ndarray]: Sorted data grouped by labels in the given order.
    """
    sorted_data = [[] for _ in range(10)]
    for img, label in dataset:
        sorted_data[label].append(np.array(img.numpy(), dtype=np.float32))

    return [np.array(sorted_data[i], dtype=np.float32) for i in order]

def save_sorted_data(data: List[np.ndarray], save_dir: str):
    """
    Save the sorted data to the specified directory.

    Args:
        data (List[np.ndarray]): Sorted data to save.
        save_dir (str): Directory where the sorted data will be stored.
    """
    os.makedirs(save_dir, exist_ok=True)
    for i, group in enumerate(data):
        save_path = os.path.join(save_dir, f"group_{i}.npy")
        np.save(save_path, group)

def generate_random_order(dataset: Dataset, save_dir: str):
    """
    Shuffle the MNIST dataset randomly and save.

    Args:
        dataset (Dataset): The MNIST dataset to shuffle.
        save_dir (str): Directory where the shuffled data will be stored.
    """
    shuffled_data = [(np.array(img.numpy(), dtype=np.float32), label) for img, label in dataset]
    random.shuffle(shuffled_data)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "random.npy")
    np.save(save_path, shuffled_data)

if __name__ == "__main__":
    # Paths and configurations
    DATA_DIR = "../data/"
    SORTED_DIR = os.path.join(DATA_DIR, "sorted/")
    RANDOM_DIR = os.path.join(DATA_DIR, "random/")

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)

    # Define orders
    ascending_order = list(range(10))
    descending_order = list(range(9, -1, -1))

    # Sort and save datasets
    sorted_data_ascending = sort_mnist_data(mnist_train, ascending_order)
    save_sorted_data(sorted_data_ascending, os.path.join(SORTED_DIR, "ascending"))

    sorted_data_descending = sort_mnist_data(mnist_train, descending_order)
    save_sorted_data(sorted_data_descending, os.path.join(SORTED_DIR, "descending"))

    # Generate and save random order
    generate_random_order(mnist_train, RANDOM_DIR)
