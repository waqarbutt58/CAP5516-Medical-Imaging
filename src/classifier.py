import os
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt


class BUSIClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def train_classifier(model, dataloaders, n_epochs=50, lr=1e-4, device="cuda", save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_acc = 0
    history = {"train_loss": [], "val_acc": [], "val_auc": []}

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0

        for images, _, labels in dataloaders["train"]:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for images, _, labels in dataloaders["val"]:
                images = images.to(device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                all_probs.extend(probs)

        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
        except ValueError:
            auc = 0.0

        history["train_loss"].append(epoch_loss / len(dataloaders["train"]))
        history["val_acc"].append(acc)
        history["val_auc"].append(auc)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(save_dir, "classifier_best.pth"))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Loss: {history['train_loss'][-1]:.4f} | "
                  f"Val Acc: {acc:.4f} | AUC: {auc:.4f}")

    print(f"Best Val Acc: {best_acc:.4f}")
    return history


def evaluate_classifier(model, dataloader, device="cuda"):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, _, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    print(classification_report(all_labels, all_preds,
                                 target_names=["Normal", "Benign", "Malignant"]))
    auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    print(f"AUC (OvR): {auc:.4f}")
    return all_labels, all_preds, all_probs


def plot_classification_results(history, save_path="results/classifier_training.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"])
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["val_acc"], color="green")
    axes[1].set_title("Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["val_auc"], color="orange")
    axes[2].set_title("Val AUC")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved: {save_path}")
