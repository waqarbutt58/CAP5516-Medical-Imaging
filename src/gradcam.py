import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, class_idx):
        self.model.eval()
        output = self.model(image_tensor)

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam, size=image_tensor.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def visualise(self, image_tensor, class_idx, class_name, save_path):
        cam = self.generate(image_tensor, class_idx)
        image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image[:, :, 0], cmap="gray")
        axes[0].set_title("Original Image")
        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title(f"Grad-CAM ({class_name})")
        axes[2].imshow(image[:, :, 0], cmap="gray", alpha=0.6)
        axes[2].imshow(cam, cmap="jet", alpha=0.4)
        axes[2].set_title("Overlay")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()


def generate_gradcam_samples(model, dataloader, device="cuda",
                              save_dir="results/gradcam_samples", n_per_class=5):
    """Generate Grad-CAM visualisations for sample images from each class."""
    from src.classifier import BUSIClassifier
    os.makedirs(save_dir, exist_ok=True)

    target_layer = model.backbone.layer4[-1].conv3
    gcam = GradCAM(model, target_layer)

    label_names = {0: "Normal", 1: "Benign", 2: "Malignant"}
    counts = {0: 0, 1: 0, 2: 0}

    model.eval()
    for images, _, labels in dataloader:
        for i in range(len(images)):
            lbl = labels[i].item()
            if counts[lbl] >= n_per_class:
                continue
            img_tensor = images[i:i+1].to(device)
            save_path = os.path.join(
                save_dir, f"{label_names[lbl].lower()}_{counts[lbl]:02d}.png"
            )
            gcam.visualise(img_tensor, lbl, label_names[lbl], save_path)
            counts[lbl] += 1

        if all(v >= n_per_class for v in counts.values()):
            break

    print(f"Grad-CAM samples saved to: {save_dir}")
