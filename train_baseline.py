"""
================================================
  Medical Image Computing — Self Study Project
================================================
  Title      : Multi-task Medical Image Analysis with
               Diffusion-Based Synthetic Augmentation
  Student    : Waqar Rauf Butt
  Roll No    : PHDAIF25M003
  Supervisor : Dr. Muhammad Farooq
  Course     : Medical Image Computing
================================================

Stage 2: Train U-Net segmentation and ResNet-50 classifier baselines.
Saves checkpoints to models/ and training curves to results/.

Usage:
    python train_baseline.py
    python train_baseline.py --epochs 50 --batch 16 --lr 0.0001
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.dataset import create_dataloaders
from src.unet import UNet
from src.train_segmentation import train_unet, plot_segmentation_results, dice_score
from src.classifier import BUSIClassifier, train_classifier, plot_classification_results, evaluate_classifier

DATA_DIR   = "data/BUSI"
MODELS_DIR = "models"
RESULTS_DIR = "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int,   default=50,    help="Number of training epochs")
    parser.add_argument("--batch",     type=int,   default=16,    help="Batch size")
    parser.add_argument("--lr",        type=float, default=1e-4,  help="Learning rate")
    parser.add_argument("--img_size",  type=int,   default=256,   help="Image size")
    parser.add_argument("--skip_unet", action="store_true",       help="Skip U-Net training")
    parser.add_argument("--skip_clf",  action="store_true",       help="Skip classifier training")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Epochs : {args.epochs} | Batch: {args.batch} | LR: {args.lr}")
    print()

    # ── Data ────────────────────────────────────────────────────────────────
    print("Loading BUSI dataset...")
    loaders = create_dataloaders(DATA_DIR, img_size=args.img_size, batch_size=args.batch)
    print(f"  Train: {len(loaders['train'].dataset)} | "
          f"Val: {len(loaders['val'].dataset)} | "
          f"Test: {len(loaders['test'].dataset)}")

    # ── U-Net ────────────────────────────────────────────────────────────────
    if not args.skip_unet:
        print("\n" + "="*55)
        print("  STAGE 1: U-Net Segmentation Baseline")
        print("="*55)
        unet = UNet(in_channels=3, out_channels=1)
        seg_history = train_unet(
            unet, loaders,
            n_epochs=args.epochs, lr=args.lr,
            device=DEVICE, save_dir=MODELS_DIR
        )
        plot_segmentation_results(seg_history, save_path=f"{RESULTS_DIR}/unet_training.png")

        # Test evaluation
        print("\nEvaluating U-Net on test set...")
        unet.load_state_dict(torch.load(f"{MODELS_DIR}/unet_best.pth", map_location=DEVICE))
        unet = unet.to(DEVICE)
        unet.eval()
        test_dice = 0
        import torch.nn as nn
        bce = nn.BCELoss()
        with torch.no_grad():
            for imgs, masks, _ in loaders["test"]:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = unet(imgs)
                test_dice += dice_score(preds, masks).item()
        test_dice /= len(loaders["test"])
        print(f"  Test Dice Score: {test_dice:.4f}")

    # ── ResNet Classifier ────────────────────────────────────────────────────
    if not args.skip_clf:
        print("\n" + "="*55)
        print("  STAGE 2: ResNet-50 Classifier Baseline")
        print("="*55)
        clf = BUSIClassifier(num_classes=3, pretrained=True)
        cls_history = train_classifier(
            clf, loaders,
            n_epochs=args.epochs, lr=args.lr,
            device=DEVICE, save_dir=MODELS_DIR
        )
        plot_classification_results(cls_history, save_path=f"{RESULTS_DIR}/classifier_training.png")

        # Test evaluation
        print("\nEvaluating classifier on test set...")
        clf.load_state_dict(torch.load(f"{MODELS_DIR}/classifier_best.pth", map_location=DEVICE))
        clf = clf.to(DEVICE)
        evaluate_classifier(clf, loaders["test"], device=DEVICE)

    print("\nBaseline training complete.")
    print(f"  Checkpoints : {MODELS_DIR}/")
    print(f"  Curves      : {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
