# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import numpy as np

from preprocess import load_mnist_data
from constants import DATA_DIR, MODEL_DIR, TRAINING_LOG_DIR, SNAPSHOT_DIR

class DenseNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super().__init__()
        layers = []
        layer_sizes = [input_size] + list(hidden_layers) + [output_size]
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2: # No ReLU after the last layer
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten the input
        return self.layers(x)


def calculate_test_accuracy(model, test_loader):
    """Calculates test accuracy for each digit class."""
    model.eval() # Set model to evaluation mode
    correct_counts = {i: 0 for i in range(10)}
    total_counts = {i: 0 for i in range(10)}
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(targets)):
                label = targets[i].item()
                total_counts[label] += 1
                if predicted[i] == targets[i]:
                    correct_counts[label] += 1

    accuracies = {}
    for label in range(10):
        accuracy = correct_counts[label] / total_counts[label] if total_counts[label] > 0 else 0
        accuracies[label] = accuracy
    return accuracies


def train_model(dataset_type, hidden_layers, epochs, batch_size, learning_rate, output_dir):
    """Trains a DenseNN model, saves snapshots, and logs training progress."""

    # 1. Load Datasets
    if dataset_type == 'shuffled':
        train_dataset = torch.load(os.path.join(DATA_DIR, 'random', 'shuffled_train_dataset.pt'))
        dataset_name = "Shuffled"
    elif dataset_type == 'increasing':
        train_dataset = torch.load(os.path.join(DATA_DIR, 'sorted', 'sorted_increasing_dataset.pt'))
        dataset_name = "Increasing"
    elif dataset_type == 'decreasing':
        train_dataset = torch.load(os.path.join(DATA_DIR, 'sorted', 'sorted_decreasing_dataset.pt'))
        dataset_name = "Decreasing"
    elif dataset_type == 'custom':
        train_dataset = torch.load(os.path.join(DATA_DIR, 'sorted', 'sorted_custom_dataset.pt'))
        dataset_name = "Custom"
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    _, test_dataset = load_mnist_data() # Load test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 2. Define Model, Loss, Optimizer
    input_size = 28 * 28
    output_size = 10
    model = DenseNN(input_size, hidden_layers, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 3. Create directories for saving models and logs
    model_dir = os.path.join(output_dir, f"model_{dataset_type}")
    snapshots_dir = os.path.join(model_dir, 'snapshots')
    os.makedirs(snapshots_dir, exist_ok=True)
    os.makedirs(TRAINING_LOG_DIR, exist_ok=True)
    log_filepath = os.path.join(TRAINING_LOG_DIR, f"training_log_{dataset_type}.npy")
    training_log = [] # List to store training data (step, epoch, batch, accuracy_dict)
    gradient_steps = 0

    # 4. Training Loop and Snapshot Saving
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch+1}/{epochs}]") # Progress bar
        for batch_idx, (data, targets) in progress_bar:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            gradient_steps += 1

            if batch_idx % 100 == 0: # Evaluate and save snapshot every 100 batches
                test_accuracy = calculate_test_accuracy(model, test_loader)
                training_log.append({'step': gradient_steps, 'epoch': epoch + 1, 'batch': batch_idx + 1, 'accuracy': test_accuracy})

                snapshot_name = f"model_{dataset_type}_epoch{epoch+1}_batch{batch_idx+1}_step{gradient_steps}.pth"
                snapshot_path = os.path.join(snapshots_dir, snapshot_name)
                torch.save(model.state_dict(), snapshot_path)

                avg_accuracy = np.mean(list(test_accuracy.values())) # Average accuracy for progress bar
                progress_bar.set_postfix({'loss': loss.item(), 'avg_accuracy': f'{avg_accuracy:.4f}'}) # Update progress bar

    np.save(log_filepath, training_log) # Save training log to file
    print(f"Finished Training for {dataset_name} dataset. Training log saved to {log_filepath}")
    final_model_path = os.path.join(model_dir, f"model_{dataset_type}_final.pth") # Save final model
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a DenseNN on MNIST with different data sorting.")
    parser.add_argument('--dataset_type', type=str, default='shuffled', choices=['shuffled', 'increasing', 'decreasing', 'custom'],
                        help='Type of dataset sorting')
    parser.add_argument('--hidden_layers', type=int, nargs='*', default=[],
                        help='Sizes of hidden layers')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs') # Reduced epochs for faster training
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default=MODEL_DIR, help='Directory to save models')

    args = parser.parse_args()
    train_model(args.dataset_type, tuple(args.hidden_layers), args.epochs, args.batch_size, args.learning_rate, args.output_dir)