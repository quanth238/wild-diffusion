import os
import subprocess
import sys
import torchvision
from PIL import Image

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    root = os.path.join(repo_root, "datasets")
    os.makedirs(root, exist_ok=True)

    print("[INFO] Downloading MNIST for FID Reference...")
    ds = torchvision.datasets.MNIST(root=root, train=False, download=True)

    out_dir = os.path.join(repo_root, "toy_outputs", "fid_ref_images")
    os.makedirs(out_dir, exist_ok=True)

    num_to_export = 5000
    print(f"[INFO] Saving {num_to_export} MNIST test images to {out_dir}")

    for i in range(num_to_export):
        img, _ = ds[i]
        path = os.path.join(out_dir, f"{i:05d}.png")
        img.save(path)

    print("[OK] Saved images.")

    npz_path = os.path.join(repo_root, "toy_outputs", "mnist_ref.npz")
    print(f"[INFO] Computing reference NPZ via fid.py into {npz_path}...")
    ref_env = os.environ.copy()
    ref_env.setdefault("MASTER_ADDR", "127.0.0.1")
    ref_env.setdefault("MASTER_PORT", "29501")
    ref_env.setdefault("RANK", "0")
    ref_env.setdefault("LOCAL_RANK", "0")
    ref_env.setdefault("WORLD_SIZE", "1")

    subprocess.run(
        [
            sys.executable,
            os.path.join(repo_root, "fid.py"), "ref",
            "--data", out_dir,
            "--dest", npz_path
        ],
        check=True,
        cwd=repo_root,
        env=ref_env,
    )
    print(f"[OK] Reference subset calculated and saved to {npz_path}")

if __name__ == "__main__":
    main()
