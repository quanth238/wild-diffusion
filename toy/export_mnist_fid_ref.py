import os
import subprocess
import torchvision
from PIL import Image

def main():
    root = "datasets"
    os.makedirs(root, exist_ok=True)

    print("[INFO] Downloading MNIST for FID Reference...")
    ds = torchvision.datasets.MNIST(root=root, train=False, download=True)

    out_dir = "toy_outputs/fid_ref_images"
    os.makedirs(out_dir, exist_ok=True)

    num_to_export = 5000
    print(f"[INFO] Saving {num_to_export} MNIST test images to {out_dir}")

    for i in range(num_to_export):
        img, _ = ds[i]
        path = os.path.join(out_dir, f"{i:05d}.png")
        img.save(path)

    print("[OK] Saved images.")

    npz_path = "toy_outputs/mnist_ref.npz"
    print(f"[INFO] Computing reference NPZ via fid.py into {npz_path}...")

    subprocess.run(
        [
            "torchrun", "--standalone", "--nproc_per_node=1",
            "fid.py", "ref",
            "--data", out_dir,
            "--dest", npz_path
        ],
        check=True
    )
    print(f"[OK] Reference subset calculated and saved to {npz_path}")

if __name__ == "__main__":
    main()
