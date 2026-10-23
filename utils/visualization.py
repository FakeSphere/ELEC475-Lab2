"""
ELEC 475: Lab 2 - Plotting & Visualization Utilities
"""
import os
import matplotlib.pyplot as plt
from typing import List, Dict

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str = "./plots/training_curves.png",
    model_name: str = "Model"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, val_losses, 'r--s', label='Val Loss')
    plt.title(f"{model_name.upper()} - Cross-Entropy Loss", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-o', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'g--^', label='Val Accuracy')
    plt.title(f"{model_name.upper()} - Top-1 Accuracy (%)", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Accuracy (%)", fontsize=10)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Training curves successfully saved to: {save_path}")

def plot_ablation_summary(
    ablation_results: Dict[str, Dict[str, List[float]]],
    save_path: str = "./plots/ablation_summary.png"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    styles = ['-', '--', '-.', ':', '-', '--', '-.']

    for idx, (model_label, data) in enumerate(ablation_results.items()):
        val_accs = data.get("val_accs", [])
        epochs = range(1, len(val_accs) + 1)
        color = colors[idx % len(colors)]
        style = styles[idx % len(styles)]
        plt.plot(epochs, val_accs, linestyle=style, marker='o', color=color, label=model_label, linewidth=2)

    plt.title("ELEC 475: Master Ablation Study (Validation Accuracy)", fontsize=13)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Validation Accuracy (%)", fontsize=11)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Master ablation summary plot saved to: {save_path}")
    
# ADDED: Command-line entry point using the recorded CIFAR-10 results.
# These are per-epoch validation accuracies, not placeholder values.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot Lab 2 ablation results.")
    parser.add_argument(
        "--plot-ablation",
        action="store_true",
        help="Generate the CIFAR-10 validation accuracy comparison."
    )
    args = parser.parse_args()

    if args.plot_ablation:
        ablation_results = {
            "LeNet (5 epochs, no aug)": {
                "val_accs": [
                    46.67, 50.32, 51.50, 52.99, 54.35
                ]
            },
            "Modern LeNet (5 epochs, no aug)": {
                "val_accs": [
                    53.26, 59.65, 61.39, 64.74, 65.47
                ]
            },
            "AlexNet: None (10 epochs, no aug)": {
                "val_accs": [
                    47.72, 57.04, 61.86, 63.40, 68.34,
                    69.69, 71.09, 72.14, 72.03, 73.74
                ]
            },
            "AlexNet: LRN (10 epochs, no aug)": {
                "val_accs": [
                    46.39, 55.87, 59.49, 64.64, 67.61,
                    70.10, 71.06, 72.68, 74.23, 73.50
                ]
            },
            "AlexNet: BN (15 epochs, no aug)": {
                "val_accs": [
                    46.25, 52.62, 65.09, 72.94, 73.35,
                    67.91, 76.62, 75.22, 78.32, 77.74,
                    78.69, 75.93, 81.40, 78.84, 82.22
                ]
            },
            "AlexNet: BN + augmentation (15 epochs)": {
                "val_accs": [
                    56.00, 61.12, 65.98, 68.65, 71.92,
                    72.49, 73.37, 76.28, 78.79, 81.34,
                    81.53, 76.85, 80.21, 82.25, 81.42
                ]
            },
            "Torchvision AlexNet (10 epochs, no aug)": {
                "val_accs": [
                    35.71, 47.30, 53.38, 58.39, 58.52,
                    61.52, 62.67, 63.46, 63.69, 64.69
                ]
            },
        }

        plot_ablation_summary(ablation_results)
    else:
        parser.print_help()