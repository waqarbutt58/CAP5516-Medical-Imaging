import os
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.dataset import BUSIDataset, load_busi_dataset, create_dataloaders
from src.classifier import BUSIClassifier, train_classifier
from src.unet import UNet
from src.train_segmentation import train_unet

SYNTHETIC_RATIOS = [0.0, 0.25, 0.50, 0.75, 1.0]


def create_augmented_loaders(real_data_dir, synthetic_dir,
                              ratio: float, batch_size=16, img_size=256, seed=42):
    """Combine real and synthetic data at a given synthetic ratio."""
    real_imgs, real_masks, real_labels = load_busi_dataset(real_data_dir)

    syn_imgs, syn_labels = [], []
    for class_idx, class_name in enumerate(["normal", "benign", "malignant"]):
        class_dir = os.path.join(synthetic_dir, class_name)
        if os.path.exists(class_dir):
            files = [os.path.join(class_dir, f)
                     for f in os.listdir(class_dir) if f.endswith(".png")]
            n_syn = int(len(real_imgs) * ratio / 3)
            random.seed(seed)
            selected = random.sample(files, min(n_syn, len(files)))
            syn_imgs.extend(selected)
            syn_labels.extend([class_idx] * len(selected))

    all_imgs = real_imgs + syn_imgs
    all_masks = real_masks + [None] * len(syn_imgs)
    all_labels = real_labels + syn_labels

    print(f"  Dataset: {len(real_imgs)} real + {len(syn_imgs)} synthetic = {len(all_imgs)} total")

    dataset = BUSIDataset(all_imgs, all_masks, all_labels, img_size=img_size, augment=True)

    # Simple 80/20 split for augmented training (test set always stays real-only)
    n = len(dataset)
    n_train = int(n * 0.8)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n - n_train],
        generator=torch.Generator().manual_seed(seed)
    )

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2),
    }


def run_ablation(real_data_dir, synthetic_dir, device="cuda", save_dir="models"):
    """Run classifier training for all synthetic ratios and record results."""
    os.makedirs(save_dir, exist_ok=True)
    results = {}

    for ratio in SYNTHETIC_RATIOS:
        print(f"\n{'='*50}")
        print(f"Synthetic ratio: {ratio:.0%}")
        print("=" * 50)

        loaders = create_augmented_loaders(real_data_dir, synthetic_dir, ratio)

        clf = BUSIClassifier(num_classes=3)
        history = train_classifier(clf, loaders, n_epochs=50, device=device, save_dir=save_dir)

        results[ratio] = {
            "best_val_auc": max(history["val_auc"]),
            "best_val_acc": max(history["val_acc"]),
            "history": history,
        }
        torch.save(clf.state_dict(),
                   os.path.join(save_dir, f"classifier_ratio_{ratio:.2f}.pth"))

    return results


def plot_ablation_results(results: dict, save_path: str = "results/ablation.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ratios = list(results.keys())
    aucs = [results[r]["best_val_auc"] for r in ratios]
    accs = [results[r]["best_val_acc"] for r in ratios]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot([f"{r:.0%}" for r in ratios], aucs, "bo-", linewidth=2, markersize=8)
    axes[0].axhline(aucs[0], color="red", linestyle="--", label="Baseline (0%)")
    axes[0].set_title("Val AUC vs Synthetic Data Ratio")
    axes[0].set_xlabel("Synthetic Data Ratio")
    axes[0].set_ylabel("AUC (OvR)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot([f"{r:.0%}" for r in ratios], accs, "go-", linewidth=2, markersize=8)
    axes[1].axhline(accs[0], color="red", linestyle="--", label="Baseline (0%)")
    axes[1].set_title("Val Accuracy vs Synthetic Data Ratio")
    axes[1].set_xlabel("Synthetic Data Ratio")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Ablation plot saved: {save_path}")
