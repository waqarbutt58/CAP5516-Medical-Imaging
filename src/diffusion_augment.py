import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from PIL import Image

CLASS_PROMPTS = {
    "normal":    "ultrasound image of a normal breast, no lesion, grayscale",
    "benign":    "ultrasound image of a benign breast lesion, well-defined edges, grayscale",
    "malignant": "ultrasound image of a malignant breast tumour, irregular edges, grayscale",
}

LABEL_TO_PROMPT = {
    0: CLASS_PROMPTS["normal"],
    1: CLASS_PROMPTS["benign"],
    2: CLASS_PROMPTS["malignant"],
}

MODEL_ID = "runwayml/stable-diffusion-v1-5"


def fine_tune_sd_on_busi(data_dir: str, output_dir: str,
                          n_epochs: int = 20, lr: float = 1e-5,
                          batch_size: int = 4, img_size: int = 512):
    """Fine-tune Stable Diffusion v1-5 U-Net on BUSI using denoising loss."""
    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from torch.cuda.amp import GradScaler, autocast
    from src.dataset import load_busi_dataset, BUSIDataset
    from torch.utils.data import DataLoader

    print(f"Loading {MODEL_ID} ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        safety_checker=None,
    )
    pipe = pipe.to("cuda")

    # Freeze VAE + text encoder — only train U-Net
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(True)

    # Enable memory-efficient attention if xformers available
    try:
        pipe.unet.enable_xformers_memory_efficient_attention()
        print("  xformers memory-efficient attention enabled.")
    except Exception:
        pass

    optimizer = AdamW(pipe.unet.parameters(), lr=lr)
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    scaler = GradScaler()

    imgs, masks, labels = load_busi_dataset(data_dir)
    dataset = BUSIDataset(imgs, masks, labels, img_size=img_size, augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)

    print(f"Fine-tuning on {len(dataset)} images for {n_epochs} epochs...")
    pipe.unet.train()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for step, (batch_imgs, _, batch_labels) in enumerate(loader):
            batch_imgs = batch_imgs.to("cuda")
            prompts = [LABEL_TO_PROMPT[l.item()] for l in batch_labels]

            # Encode images → latent space
            with torch.no_grad():
                latents = pipe.vae.encode(batch_imgs).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device="cuda"
            ).long()
            noisy_lat = noise_scheduler.add_noise(latents, noise, timesteps)

            # Encode text prompts
            with torch.no_grad():
                tokens = pipe.tokenizer(
                    prompts, return_tensors="pt",
                    padding="max_length",
                    max_length=pipe.tokenizer.model_max_length,
                    truncation=True
                ).to("cuda")
                enc_text = pipe.text_encoder(**tokens).last_hidden_state

            # Predict noise with mixed precision
            with autocast():
                noise_pred = pipe.unet(
                    noisy_lat, timesteps,
                    encoder_hidden_states=enc_text
                ).sample
                loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    pipe.save_pretrained(output_dir)
    print(f"Fine-tuned model saved: {output_dir}")
    return pipe


def generate_synthetic_images(pipe_dir: str, n_per_class: int = 150,
                               output_dir: str = "data/synthetic",
                               guidance_scale: float = 7.5,
                               num_inference_steps: int = 30):
    """Generate synthetic BUSI images using fine-tuned Stable Diffusion."""
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        pipe_dir,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")

    for class_name, prompt in CLASS_PROMPTS.items():
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        print(f"  Generating {n_per_class} images for [{class_name}] ...")
        for i in range(n_per_class):
            g = torch.Generator("cuda").manual_seed(i * 100 + hash(class_name) % 10000)
            result = pipe(
                prompt,
                generator=g,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=256, width=256,
            )
            img = result.images[0].convert("L")  # grayscale like real BUSI
            img.save(os.path.join(class_dir, f"synthetic_{i:04d}.png"))

    print(f"Synthetic images saved to: {output_dir}")


def compute_fid(real_dir: str, synthetic_dir: str, device: str = "cuda"):
    """Compute FID score between real and synthetic images."""
    try:
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "pytorch_fid", real_dir, synthetic_dir,
             "--device", device],
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print("FID error:", result.stderr[:300])
        return result.stdout
    except Exception as e:
        print(f"FID computation failed: {e}")
        return None
