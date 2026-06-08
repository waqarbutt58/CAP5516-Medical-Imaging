import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from PIL import Image

CLASS_PROMPTS = {
    "normal":    "ultrasound image of a normal breast, no lesion, grayscale",
    "benign":    "ultrasound image of a benign breast lesion, well-defined edges, grayscale",
    "malignant": "ultrasound image of a malignant breast tumour, irregular edges, grayscale",
}

LABEL_TO_PROMPT = {0: CLASS_PROMPTS["normal"],
                   1: CLASS_PROMPTS["benign"],
                   2: CLASS_PROMPTS["malignant"]}


def fine_tune_sd_on_busi(data_dir: str, output_dir: str,
                          n_epochs: int = 20, lr: float = 1e-5,
                          batch_size: int = 4, img_size: int = 512):
    """Fine-tune Stable Diffusion v1-5 U-Net on BUSI using denoising loss."""
    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from src.dataset import load_busi_dataset, BUSIDataset
    from torch.utils.data import DataLoader

    model_id = "runwayml/stable-diffusion-v1-5"
    print(f"Loading {model_id} ...")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to("cuda")

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(True)

    optimizer = AdamW(pipe.unet.parameters(), lr=lr)
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    scaler = GradScaler()

    imgs, masks, labels = load_busi_dataset(data_dir)
    dataset = BUSIDataset(imgs, masks, labels, img_size=img_size, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    print(f"Fine-tuning on {len(dataset)} images for {n_epochs} epochs...")
    pipe.unet.train()

    for epoch in range(n_epochs):
        epoch_loss = 0
        for batch_imgs, _, batch_labels in loader:
            batch_imgs = batch_imgs.to("cuda")
            prompts = [LABEL_TO_PROMPT[l.item()] for l in batch_labels]

            with torch.no_grad():
                latents = pipe.vae.encode(batch_imgs).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (latents.shape[0],), device="cuda"
            ).long()
            noisy_lat = scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                tokens = pipe.tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True
                ).to("cuda")
                enc_text = pipe.text_encoder(**tokens).last_hidden_state

            with autocast():
                noise_pred = pipe.unet(noisy_lat, timesteps, enc_text).sample
                loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {epoch_loss/len(loader):.4f}")

    os.makedirs(output_dir, exist_ok=True)
    pipe.save_pretrained(output_dir)
    print(f"Fine-tuned model saved: {output_dir}")
    return pipe


def generate_synthetic_images(pipe_dir: str, n_per_class: int = 100,
                               output_dir: str = "data/synthetic",
                               guidance_scale: float = 7.5,
                               num_inference_steps: int = 30):
    """Generate synthetic BUSI images using fine-tuned Stable Diffusion."""
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        pipe_dir, torch_dtype=torch.float16
    ).to("cuda")
    pipe.safety_checker = None  # Disable for medical images

    for class_name, prompt in CLASS_PROMPTS.items():
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        print(f"Generating {n_per_class} images for class: {class_name}")
        for i in range(n_per_class):
            g = torch.Generator("cuda").manual_seed(i * 100)
            img = pipe(
                prompt, generator=g,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            img = img.convert("L")
            img.save(os.path.join(class_dir, f"synthetic_{i:04d}.png"))

    print(f"Synthetic images saved to: {output_dir}")


def compute_fid(real_dir: str, synthetic_dir: str):
    """Compute FID score between real and synthetic images."""
    try:
        import subprocess
        result = subprocess.run(
            ["python", "-m", "pytorch_fid", real_dir, synthetic_dir],
            capture_output=True, text=True
        )
        print(result.stdout)
        return result.stdout
    except Exception as e:
        print(f"FID computation failed: {e}")
        return None
