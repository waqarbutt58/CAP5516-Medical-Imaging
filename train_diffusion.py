"""
Stage 3: Fine-tune Stable Diffusion v1-5 on BUSI and generate synthetic images.

Steps:
  1. Fine-tune SD U-Net on BUSI images (class-conditioned via text prompt)
  2. Generate N synthetic images per class
  3. Compute FID score (real vs synthetic)

Usage:
    python train_diffusion.py --epochs 20 --n_per_class 150
    python train_diffusion.py --skip_finetune   # generate only (if SD already fine-tuned)
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR      = "data/BUSI"
SD_OUTPUT_DIR = "models/finetuned_sd"
SYNTHETIC_DIR = "data/synthetic"
RESULTS_DIR   = "results/synthetic_samples"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


def install_diffusers():
    """Ensure diffusers & accelerate are installed."""
    import subprocess
    pkgs = ["diffusers", "transformers", "accelerate", "safetensors"]
    for pkg in pkgs:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


def finetune(epochs, batch_size, lr, img_size):
    print("\n" + "="*55)
    print("  STAGE 3A: Fine-tuning Stable Diffusion on BUSI")
    print("="*55)
    from src.diffusion_augment import fine_tune_sd_on_busi
    fine_tune_sd_on_busi(
        data_dir=DATA_DIR,
        output_dir=SD_OUTPUT_DIR,
        n_epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        img_size=img_size,
    )


def generate(n_per_class, guidance_scale, steps):
    print("\n" + "="*55)
    print("  STAGE 3B: Generating Synthetic Images")
    print("="*55)
    from src.diffusion_augment import generate_synthetic_images
    generate_synthetic_images(
        pipe_dir=SD_OUTPUT_DIR,
        n_per_class=n_per_class,
        output_dir=SYNTHETIC_DIR,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
    )

    # Save sample grid
    save_sample_grid()


def save_sample_grid(n_per_class=5):
    """Save a visual grid of synthetic samples for inspection."""
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np

    os.makedirs(RESULTS_DIR, exist_ok=True)
    classes = ["normal", "benign", "malignant"]
    fig, axes = plt.subplots(len(classes), n_per_class,
                              figsize=(n_per_class * 3, len(classes) * 3))

    for r, cls in enumerate(classes):
        cls_dir = os.path.join(SYNTHETIC_DIR, cls)
        files = sorted([f for f in os.listdir(cls_dir) if f.endswith(".png")])[:n_per_class]
        for c, fname in enumerate(files):
            img = np.array(Image.open(os.path.join(cls_dir, fname)).convert("L"))
            axes[r, c].imshow(img, cmap="gray")
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_ylabel(cls.capitalize(), fontsize=12, rotation=90, labelpad=10)

    plt.suptitle("Synthetic BUSI Samples (per class)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "synthetic_grid.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sample grid saved: {save_path}")


def compute_fid_score():
    """Compute FID between real and synthetic images."""
    print("\n" + "="*55)
    print("  STAGE 3C: Computing FID Score")
    print("="*55)
    try:
        import subprocess
        # Combine all real images into one temp folder
        import shutil, tempfile
        real_tmp = tempfile.mkdtemp()
        syn_tmp  = tempfile.mkdtemp()

        for cls in ["normal", "benign", "malignant"]:
            for fname in os.listdir(os.path.join(DATA_DIR, cls)):
                if not fname.endswith("_mask.png") and fname.endswith(".png"):
                    shutil.copy(os.path.join(DATA_DIR, cls, fname),
                                os.path.join(real_tmp, f"{cls}_{fname}"))
            for fname in os.listdir(os.path.join(SYNTHETIC_DIR, cls)):
                if fname.endswith(".png"):
                    shutil.copy(os.path.join(SYNTHETIC_DIR, cls, fname),
                                os.path.join(syn_tmp, f"{cls}_{fname}"))

        result = subprocess.run(
            [sys.executable, "-m", "pytorch_fid", real_tmp, syn_tmp, "--device", DEVICE],
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print("FID stderr:", result.stderr[:300])

        shutil.rmtree(real_tmp)
        shutil.rmtree(syn_tmp)
    except Exception as e:
        print(f"FID computation skipped: {e}")
        print("Install with: pip install pytorch-fid")


def main():
    parser = argparse.ArgumentParser(description="Diffusion Augmentation Pipeline")
    parser.add_argument("--epochs",       type=int,   default=20,   help="Fine-tune epochs")
    parser.add_argument("--batch",        type=int,   default=4,    help="Batch size (keep low, SD is large)")
    parser.add_argument("--lr",           type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--img_size",     type=int,   default=512,  help="Training image size for SD")
    parser.add_argument("--n_per_class",  type=int,   default=150,  help="Synthetic images to generate per class")
    parser.add_argument("--guidance",     type=float, default=7.5,  help="Classifier-free guidance scale")
    parser.add_argument("--steps",        type=int,   default=30,   help="Denoising inference steps")
    parser.add_argument("--skip_finetune",action="store_true",      help="Skip fine-tuning (use existing SD)")
    parser.add_argument("--skip_generate",action="store_true",      help="Skip image generation")
    parser.add_argument("--fid",          action="store_true",      help="Compute FID score after generation")
    args = parser.parse_args()

    print(f"Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    install_diffusers()

    if not args.skip_finetune:
        finetune(args.epochs, args.batch, args.lr, args.img_size)

    if not args.skip_generate:
        generate(args.n_per_class, args.guidance, args.steps)

    if args.fid:
        compute_fid_score()

    print("\nDiffusion stage complete.")
    print(f"  Fine-tuned SD : {SD_OUTPUT_DIR}/")
    print(f"  Synthetic data: {SYNTHETIC_DIR}/")
    print(f"  Sample grid   : {RESULTS_DIR}/synthetic_grid.png")


if __name__ == "__main__":
    main()
