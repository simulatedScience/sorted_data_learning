import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
import os

from constants import DATA_DIR

def load_mnist_data():
    """Loads MNIST training and test datasets."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) # Standard MNIST normalization
    train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def sort_data(dataset, sort_order):
    """Sorts the dataset according to the given order of labels."""
    indices = []
    for label in sort_order:
        label_indices = [i for i, target in enumerate(dataset.targets) if target == label]
        indices.extend(label_indices)
    return torch.utils.data.Subset(dataset, indices)

def shuffle_data(dataset):
    """Shuffles the dataset randomly."""
    indices = torch.randperm(len(dataset)).tolist()
    return torch.utils.data.Subset(dataset, indices)

def custom_sort_data(dataset, order_list):
    """
    Creates a custom sort order from a list of label groups.
    Example: custom_sort_data([8, [9, 6, 0], [5, 3, 2], [7, 4, 1]]) might generate
    [8,8,8, 6,9,0,0,6,9, 2,2,3,5,3,5, 7,1,4,4,7,1] as the final order.
    """
    sorted_data = []
    for group in order_list:
        # 1. get all samples with labels in the group
        if isinstance(group, int):
            group: list[int] = [group]
        group_data = []
        for label in group:
            label_indices = [i for i, target in enumerate(dataset.targets) if target == label]
            group_data.extend(label_indices)
        # 2. shuffle the samples within the group
        group_data = shuffle_data(group_data)
        # 3. add the shuffled samples to the full order
        sorted_data.extend(group_data)
    return torch.utils.data.Subset(dataset, sorted_data)

if __name__ == '__main__':
    train_dataset, _ = load_mnist_data()

    # Create 'data/sorted' and 'data/random' directories if they don't exist
    os.makedirs(os.path.join(DATA_DIR, 'sorted'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'random'), exist_ok=True)

    # 1. Randomly shuffled data
    shuffled_train_dataset = shuffle_data(train_dataset)
    torch.save(shuffled_train_dataset, os.path.join(DATA_DIR, 'random', 'shuffled_train_dataset.pt'))
    print("Randomly shuffled data saved to data/random/shuffled_train_dataset.pt")

    # 2. Sorted increasingly [0, 1, 2, ..., 9]
    increasing_order = list(range(10))
    sorted_increasing_dataset = sort_data(train_dataset, increasing_order)
    torch.save(sorted_increasing_dataset, os.path.join(DATA_DIR, 'sorted', 'sorted_increasing_dataset.pt'))
    print("Increasingly sorted data saved to data/sorted/sorted_increasing_dataset.pt")

    # 3. Sorted decreasingly [9, 8, 7, ..., 0]
    decreasing_order = list(range(9, -1, -1))
    sorted_decreasing_dataset = sort_data(train_dataset, decreasing_order)
    torch.save(sorted_decreasing_dataset, os.path.join(DATA_DIR, 'sorted', 'sorted_decreasing_dataset.pt'))
    print("Decreasingly sorted data saved to data/sorted/sorted_decreasing_dataset.pt")

    # 4. Custom order [8, random [9, 6, 0], random [5, 3, 2], random [7, 4, 1]]
    custom_order_list = [8, [9, 6, 0], [5, 3, 2], [7, 4, 1]]
    sorted_custom_dataset = custom_sort_data(train_dataset, custom_order_list)
    torch.save(sorted_custom_dataset, os.path.join(DATA_DIR, 'sorted', 'sorted_custom_dataset.pt'))
    print("Custom sorted data saved to data/sorted/sorted_custom_dataset.pt")

    print("Preprocessing complete. Datasets saved in data/sorted/ and data/random/")