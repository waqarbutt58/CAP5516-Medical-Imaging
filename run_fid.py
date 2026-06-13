import subprocess, sys, os, shutil, tempfile

DATA_DIR = "data/BUSI"
SYNTHETIC_DIR = "data/synthetic"

real_tmp = tempfile.mkdtemp()
syn_tmp  = tempfile.mkdtemp()

from PIL import Image
SIZE = (299, 299)  # Inception input size

for cls in ["normal", "benign", "malignant"]:
    for fname in os.listdir(os.path.join(DATA_DIR, cls)):
        if not fname.endswith("_mask.png") and fname.endswith(".png"):
            img = Image.open(os.path.join(DATA_DIR, cls, fname)).convert("RGB").resize(SIZE)
            img.save(os.path.join(real_tmp, cls + "_" + fname))
    for fname in os.listdir(os.path.join(SYNTHETIC_DIR, cls)):
        if fname.endswith(".png"):
            img = Image.open(os.path.join(SYNTHETIC_DIR, cls, fname)).convert("RGB").resize(SIZE)
            img.save(os.path.join(syn_tmp, cls + "_" + fname))

print(f"Real images   : {len(os.listdir(real_tmp))}")
print(f"Synthetic imgs: {len(os.listdir(syn_tmp))}")
print("Computing FID score...")
sys.stdout.flush()

result = subprocess.run(
    [sys.executable, "-m", "pytorch_fid", real_tmp, syn_tmp, "--device", "cuda"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("FID ERROR:", result.returncode)

shutil.rmtree(real_tmp)
shutil.rmtree(syn_tmp)
print("Done.")
