# src/visualize.py
import torch
import matplotlib.pyplot as plt
import os

from constants import DATA_DIR, RESULTS_DIR

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    dataset_files = {
        "Random Shuffled": os.path.join(DATA_DIR, 'random', 'shuffled_train_dataset.pt'),
        "Sorted Increasing": os.path.join(DATA_DIR, 'sorted', 'sorted_increasing_dataset.pt'),
        "Sorted Decreasing": os.path.join(DATA_DIR, 'sorted', 'sorted_decreasing_dataset.pt'),
        "Sorted Custom": os.path.join(DATA_DIR, 'sorted', 'sorted_custom_dataset.pt'),
    }

    for name, file_path in dataset_files.items():
        dataset = torch.load(file_path)
        labels = [dataset.dataset.targets[i].item() for i in dataset.indices] # Get labels in sorted order

        plt.figure(figsize=(10, 5)) # Adjust figure size for better readability
        plt.plot(range(len(labels)), labels, marker='s', linestyle='', markersize=5) # Use dots for better visualization of many points
        plt.title(f"Dataset Label Distribution: {name}")
        plt.xlabel("Index in Training Data")
        plt.ylabel("Label")
        plt.yticks(range(10)) # Show all label ticks from 0 to 9
        plt.grid(axis='y') # Add horizontal grid lines for better readability

        plot_filename = os.path.join(RESULTS_DIR, "data_visualization", f"label_distribution_{name.lower().replace(' ', '_')}.png")
        plt.savefig(plot_filename)
        print(f"Label distribution plot for {name} saved to {plot_filename}")
        plt.close() # Close plot to free memory

    print("Dataset visualization complete. Plots saved in results/")