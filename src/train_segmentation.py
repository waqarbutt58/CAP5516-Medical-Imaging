import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.unet import UNet


def dice_loss(pred, target, smooth=1):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def dice_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    return (2.0 * intersection) / (pred.sum() + target.sum() + 1e-8)


def train_unet(model, dataloaders, n_epochs=50, lr=1e-4, device="cuda", save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    bce_loss = nn.BCELoss()

    history = {"train_loss": [], "val_dice": [], "val_loss": []}
    best_dice = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0

        for images, masks, _ in dataloaders["train"]:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = bce_loss(preds, masks) + dice_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        val_dice, val_loss = 0, 0
        with torch.no_grad():
            for images, masks, _ in dataloaders["val"]:
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                val_loss += (bce_loss(preds, masks) + dice_loss(preds, masks)).item()
                val_dice += dice_score(preds, masks).item()

        val_dice /= len(dataloaders["val"])
        val_loss /= len(dataloaders["val"])
        scheduler.step(val_loss)

        history["train_loss"].append(epoch_loss / len(dataloaders["train"]))
        history["val_dice"].append(val_dice)
        history["val_loss"].append(val_loss)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(save_dir, "unet_best.pth"))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Loss: {history['train_loss'][-1]:.4f} | "
                  f"Val Dice: {val_dice:.4f}")

    print(f"Best Val Dice: {best_dice:.4f}")
    return history


def plot_segmentation_results(history, save_path="results/unet_training.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("U-Net Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["val_dice"], label="Val Dice", color="green")
    axes[1].set_title("U-Net Dice Score")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved: {save_path}")
