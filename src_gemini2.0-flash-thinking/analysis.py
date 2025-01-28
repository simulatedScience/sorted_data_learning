
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np

from constants import MODEL_DIR, RESULTS_DIR
from train import DenseNN

ACTIVATIONS_DIR = os.path.join(RESULTS_DIR, 'activations')
CONFIDENCE_IMAGES_DIR = os.path.join(RESULTS_DIR, 'confidence_images')

def generate_grouped_heatmaps(model_paths, output_dir):
    """Generates grouped heatmaps for all models in a single figure."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(len(model_paths), 10, figsize=(15, 3 * len(model_paths))) # Adjust figure size for all models
    dataset_types = ['Shuffled', 'Increasing', 'Decreasing', 'Custom'] # Agent labels

    for model_idx, (dataset_type, model_path) in enumerate(zip(dataset_types, model_paths)):
        state_dict = torch.load(model_path)
        weights = state_dict['layers.0.weight']
        input_size = int(np.sqrt(weights.shape[1]))

        for digit in range(10):
            digit_weights = weights[digit].reshape(input_size, input_size).detach().numpy()
            ax = axes[model_idx, digit] # Access subplot for current model and digit
            im = ax.imshow(digit_weights, cmap='seismic', vmin=-np.max(np.abs(digit_weights)), vmax=np.max(np.abs(digit_weights))) # Consistent color scale
            ax.set_xticks([])
            ax.set_yticks([])
            if digit == 0:
                ax.set_ylabel(dataset_type, rotation=90, labelpad=20, fontsize=12) # Agent label on the left
            if model_idx == 0:
                ax.set_title(str(digit)) # Digit label on top

    fig.suptitle("Heatmaps of Weights for Different Agents", fontsize=16) # Overall figure title
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to fit title
    heatmap_path = os.path.join(output_dir, "grouped_heatmaps.png")
    plt.savefig(heatmap_path)
    plt.close(fig) # Close figure
    print(f"Grouped heatmaps saved to {heatmap_path}")


def generate_grouped_confidence_plots(model_paths, output_dir, test_loader):
    """Generates grouped confidence plots for all models in separate figures for min, max, avg."""
    os.makedirs(output_dir, exist_ok=True)
    dataset_types = ['Shuffled', 'Increasing', 'Decreasing', 'Custom']
    plot_types = ['min', 'max', 'average']

    for plot_type in plot_types:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True) # 2x2 grid for 4 agents, shared axes
        axes = axes.flatten() # Flatten axes array for easy indexing

        for model_idx, (dataset_type, model_path) in enumerate(zip(dataset_types, model_paths)):
            state_dict = torch.load(model_path)
            input_size = 28 * 28
            output_size = 10
            model = DenseNN(input_size, [], output_size)
            model.load_state_dict(state_dict)
            model.eval()

            confidence_diffs_per_label = {i: [] for i in range(10)}
            with torch.no_grad():
                for data, targets in test_loader:
                    outputs = model(data)
                    probabilities = torch.softmax(outputs, dim=1)
                    max_probs, _ = torch.max(probabilities, dim=1)
                    sorted_probs, _ = torch.sort(probabilities, dim=1, descending=True)
                    second_max_probs = sorted_probs[:, 1]
                    confidence_diffs = (max_probs - second_max_probs).tolist()

                    for i in range(len(targets)):
                        label = targets[i].item()
                        confidence_diffs_per_label[label].append(confidence_diffs[i])

            ax = axes[model_idx]
            y_values = []
            for label in range(10):
                diffs = confidence_diffs_per_label[label]
                if plot_type == 'min':
                    y_values.append(np.min(diffs) if diffs else 0)
                elif plot_type == 'max':
                    y_values.append(np.max(diffs) if diffs else 0)
                elif plot_type == 'average':
                    y_values.append(np.mean(diffs) if diffs else 0)

            ax.bar(range(10), y_values)
            ax.set_xticks(range(10))
            ax.set_title(dataset_type) # Agent title for each subplot
            ax.set_ylim([0, 1])
            if model_idx in [2, 3]: # Only label x-axis for bottom row
                ax.set_xlabel("Digit Label")
            if model_idx in [0, 2]: # Only label y-axis for left column
                ax.set_ylabel(f"{plot_type.capitalize()} Confidence")


        fig.suptitle(f"{plot_type.capitalize()} Confidence Difference per Digit for Different Agents", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
        plot_path = os.path.join(output_dir, f"grouped_confidence_{plot_type}.png")
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Grouped confidence plot ({plot_type}) saved to {plot_path}")


if __name__ == '__main__':
    dataset_types = ['shuffled', 'increasing', 'decreasing', 'custom']
    model_paths = [os.path.join(MODEL_DIR, f"model_{dataset_type}.pth") for dataset_type in dataset_types]

    from preprocess import load_mnist_data
    _, test_dataset = load_mnist_data()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    os.makedirs(ACTIVATIONS_DIR, exist_ok=True)
    os.makedirs(CONFIDENCE_IMAGES_DIR, exist_ok=True)

    print("Generating grouped heatmaps...")
    generate_grouped_heatmaps(model_paths, ACTIVATIONS_DIR)

    print("Generating grouped confidence plots...")
    generate_grouped_confidence_plots(model_paths, CONFIDENCE_IMAGES_DIR, test_loader)

    print("Analysis complete. Grouped heatmaps and confidence plots saved in results/activations/ and results/confidence_images/")