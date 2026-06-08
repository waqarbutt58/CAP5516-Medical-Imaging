"""
CAP5516 – Multi-task Medical Image Analysis with Diffusion-Based Synthetic Augmentation
Main pipeline script. Run each stage independently or end-to-end.

Usage:
    python main.py --stage eda
    python main.py --stage baseline
    python main.py --stage diffusion
    python main.py --stage ablation
    python main.py --stage report
    python main.py --stage all
"""

import argparse
import torch
import os

DATA_DIR       = "data/BUSI"
SYNTHETIC_DIR  = "data/synthetic"
MODELS_DIR     = "models"
RESULTS_DIR    = "results"
SD_OUTPUT_DIR  = "models/finetuned_sd"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE       = 256
BATCH_SIZE     = 16
N_EPOCHS       = 50


def stage_eda():
    print("\n[Stage 1] EDA & Preprocessing")
    from src.dataset import plot_eda, create_dataloaders
    plot_eda(DATA_DIR, save_path=f"{RESULTS_DIR}/eda.png")
    loaders = create_dataloaders(DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE)
    print(f"  Train: {len(loaders['train'].dataset)} | "
          f"Val: {len(loaders['val'].dataset)} | "
          f"Test: {len(loaders['test'].dataset)}")
    return loaders


def stage_baseline(loaders=None):
    print("\n[Stage 2] Baseline Models")
    if loaders is None:
        from src.dataset import create_dataloaders
        loaders = create_dataloaders(DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE)

    # U-Net segmentation
    print("  Training U-Net segmentation baseline...")
    from src.unet import UNet
    from src.train_segmentation import train_unet, plot_segmentation_results
    unet = UNet(in_channels=3, out_channels=1)
    seg_history = train_unet(unet, loaders, n_epochs=N_EPOCHS, device=DEVICE, save_dir=MODELS_DIR)
    plot_segmentation_results(seg_history, save_path=f"{RESULTS_DIR}/unet_training.png")

    # ResNet classifier
    print("  Training ResNet-50 classifier baseline...")
    from src.classifier import BUSIClassifier, train_classifier, plot_classification_results
    clf = BUSIClassifier(num_classes=3)
    cls_history = train_classifier(clf, loaders, n_epochs=N_EPOCHS, device=DEVICE, save_dir=MODELS_DIR)
    plot_classification_results(cls_history, save_path=f"{RESULTS_DIR}/classifier_training.png")

    return seg_history, cls_history


def stage_diffusion():
    print("\n[Stage 3] Diffusion Augmentation")
    from src.diffusion_augment import fine_tune_sd_on_busi, generate_synthetic_images
    fine_tune_sd_on_busi(DATA_DIR, output_dir=SD_OUTPUT_DIR, n_epochs=20, batch_size=4)
    generate_synthetic_images(SD_OUTPUT_DIR, n_per_class=150, output_dir=SYNTHETIC_DIR)


def stage_ablation():
    print("\n[Stage 4] Augmented Training & Ablation Study")
    from src.augmented_training import run_ablation, plot_ablation_results
    results = run_ablation(DATA_DIR, SYNTHETIC_DIR, device=DEVICE, save_dir=MODELS_DIR)
    plot_ablation_results(results, save_path=f"{RESULTS_DIR}/ablation.png")
    return results


def stage_report(metrics=None):
    print("\n[Stage 5] Evaluation & Report")
    if metrics is None:
        metrics = {}

    # Grad-CAM
    from src.dataset import create_dataloaders
    from src.classifier import BUSIClassifier
    from src.gradcam import generate_gradcam_samples

    loaders = create_dataloaders(DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE)
    clf = BUSIClassifier(num_classes=3)
    ckpt = os.path.join(MODELS_DIR, "classifier_best.pth")
    if os.path.exists(ckpt):
        clf.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        clf = clf.to(DEVICE)
        generate_gradcam_samples(clf, loaders["test"], device=DEVICE,
                                  save_dir=f"{RESULTS_DIR}/gradcam_samples")

    from src.report_generator import generate_report
    generate_report(metrics, save_path=f"{RESULTS_DIR}/clinical_report.pdf")


def main():
    parser = argparse.ArgumentParser(description="CAP5516 Medical Imaging Pipeline")
    parser.add_argument("--stage", choices=["eda", "baseline", "diffusion", "ablation", "report", "all"],
                        default="eda", help="Pipeline stage to run")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    if args.stage == "eda":
        stage_eda()
    elif args.stage == "baseline":
        stage_baseline()
    elif args.stage == "diffusion":
        stage_diffusion()
    elif args.stage == "ablation":
        stage_ablation()
    elif args.stage == "report":
        stage_report()
    elif args.stage == "all":
        loaders = stage_eda()
        seg_history, cls_history = stage_baseline(loaders)
        stage_diffusion()
        ablation_results = stage_ablation()
        metrics = {
            "baseline_dice": max(seg_history["val_dice"]),
            "baseline_auc":  max(cls_history["val_auc"]),
            "baseline_acc":  max(cls_history["val_acc"]),
            "ablation":      ablation_results,
        }
        stage_report(metrics)


if __name__ == "__main__":
    main()
