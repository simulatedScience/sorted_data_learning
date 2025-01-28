# src/plot_training.py
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm # Import colormap

TRAINING_LOG_DIR = 'results/training_logs'
RESULTS_DIR = 'results'
TRAINING_PLOTS_DIR = os.path.join(RESULTS_DIR, 'training_plots')

if __name__ == '__main__':
    dataset_types = ['shuffled', 'increasing', 'decreasing', 'custom']
    os.makedirs(TRAINING_PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True) # 4 subplots vertically
    fig.suptitle("Test Accuracy During Training for Different Agents", fontsize=16)

    colors = cm.get_cmap('rainbow', 10) # Rainbow colormap for 10 digits

    for idx, dataset_type in enumerate(dataset_types):
        log_filepath = os.path.join(TRAINING_LOG_DIR, f"training_log_{dataset_type}.npy")
        training_log_data = np.load(log_filepath, allow_pickle=True)
        ax = axes[idx]
        ax.set_title(f"Agent: {dataset_type.capitalize()}")
        ax.set_ylabel("Test Accuracy")
        ax.grid(axis='x', linestyle='--', color='gray', linewidth=0.5) # Vertical grid lines

        epoch_boundaries_steps = [] # To store gradient steps at epoch boundaries
        last_epoch = 0

        for log_entry in training_log_data:
            step = log_entry['step']
            epoch = log_entry['epoch']
            accuracy_dict = log_entry['accuracy']

            if epoch != last_epoch: # Detect epoch boundary
                epoch_boundaries_steps.append(step)
                last_epoch = epoch

            for digit in range(10):
                accuracy = accuracy_dict[digit]
                ax.plot(step, accuracy, marker='o', markersize=2, linestyle='-', linewidth=0.5, color=colors(digit), label=f'Digit {digit}' if idx == 0 else None) # Label only once

        for step_val in epoch_boundaries_steps[1:]: # Add vertical lines, skip first step (start of training)
            ax.axvline(x=step_val, color='gray', linestyle='--', linewidth=0.8)

    axes[-1].set_xlabel("Gradient Steps") # X-axis label for the bottom subplot
    axes[0].legend(loc='upper right', ncol=5) # Legend in the first subplot
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    plot_filename = os.path.join(TRAINING_PLOTS_DIR, "training_accuracy_plot.png")
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"Training accuracy plot saved to {plot_filename}")
    print("Training visualization complete. Plot saved in results/training_plots/")