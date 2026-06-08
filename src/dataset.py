import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

CLASS_LABELS = {"normal": 0, "benign": 1, "malignant": 2}


class BUSIDataset(Dataset):
    def __init__(self, image_paths, mask_paths, labels, img_size=256, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.img_size = img_size

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]) if augment else None

        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.mask_paths[idx] is not None and os.path.exists(self.mask_paths[idx]):
            mask = Image.open(self.mask_paths[idx]).convert("L")
        else:
            mask = Image.fromarray(np.zeros((self.img_size, self.img_size), dtype=np.uint8))

        if self.aug_transform:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.aug_transform(image)
            torch.manual_seed(seed)
            mask = self.aug_transform(mask)

        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()

        return image, mask, torch.tensor(label, dtype=torch.long)


def load_busi_dataset(data_dir: str):
    image_paths, mask_paths, labels = [], [], []

    for class_name, class_idx in CLASS_LABELS.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        images = [f for f in os.listdir(class_dir) if not f.endswith("_mask.png") and f.endswith(".png")]

        for img_file in sorted(images):
            img_path = os.path.join(class_dir, img_file)
            mask_file = img_file.replace(".png", "_mask.png")
            mask_path = os.path.join(class_dir, mask_file)

            image_paths.append(img_path)
            mask_paths.append(mask_path if os.path.exists(mask_path) else None)
            labels.append(class_idx)

    return image_paths, mask_paths, labels


def create_dataloaders(data_dir, img_size=256, batch_size=16, seed=42):
    imgs, masks, labels = load_busi_dataset(data_dir)

    idx = list(range(len(imgs)))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=labels, random_state=seed)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50,
        stratify=[labels[i] for i in temp_idx],
        random_state=seed
    )

    def subset(indices, augment=False):
        return BUSIDataset(
            [imgs[i] for i in indices],
            [masks[i] for i in indices],
            [labels[i] for i in indices],
            img_size=img_size, augment=augment
        )

    train_ds = subset(train_idx, augment=True)
    val_ds = subset(val_idx, augment=False)
    test_ds = subset(test_idx, augment=False)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
    }


def plot_eda(data_dir: str, save_path: str = "results/eda.png"):
    imgs, masks, labels = load_busi_dataset(data_dir)
    label_names = {0: "Normal", 1: "Benign", 2: "Malignant"}
    counts = {label_names[l]: labels.count(l) for l in [0, 1, 2]}

    fig = plt.figure(figsize=(14, 4))

    ax_bar = fig.add_subplot(1, 4, 1)
    ax_bar.bar(counts.keys(), counts.values(), color=["#4caf50", "#2196f3", "#f44336"])
    ax_bar.set_title("Class Distribution")
    ax_bar.set_ylabel("Number of Images")
    for k, v in zip(counts.keys(), counts.values()):
        ax_bar.text(k, v + 2, str(v), ha="center", fontweight="bold")

    # Sample one image per class
    sample_imgs = []
    for cls in [0, 1, 2]:
        idx = next(i for i, l in enumerate(labels) if l == cls)
        sample_imgs.append((imgs[idx], label_names[cls]))

    for i, (img_path, cls_name) in enumerate(sample_imgs):
        img = np.array(Image.open(img_path).convert("L"))
        ax = fig.add_subplot(1, 4, i + 2)
        ax.imshow(img, cmap="gray")
        ax.set_title(cls_name, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"EDA plot saved: {save_path}")
