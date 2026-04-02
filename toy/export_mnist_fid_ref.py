import os
import subprocess
import sys
from PIL import Image


def _load_mnist_test_images_no_torchvision():
    """Fallback loader using internal IDX parser (no torchvision dependency)."""

    from pathlib import Path

    from toy.data_backends.provider import _load_mnist_tensors

    cache_root = Path.home() / ".cache" / "wild_diffusion" / "mnist"
    _, _, test_images, _ = _load_mnist_tensors(cache_root)
    return test_images


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    root = os.path.join(repo_root, "datasets")
    os.makedirs(root, exist_ok=True)

    ds = None
    test_images = None
    try:
        import torchvision

        print("[INFO] Loading MNIST via torchvision for FID reference...")
        ds = torchvision.datasets.MNIST(root=root, train=False, download=True)
    except Exception as exc:
        print(
            f"[WARN] torchvision MNIST loader unavailable ({exc}). "
            "Falling back to internal IDX MNIST loader.",
            flush=True,
        )
        test_images = _load_mnist_test_images_no_torchvision()

    out_dir = os.path.join(repo_root, "toy_outputs", "fid_ref_images")
    os.makedirs(out_dir, exist_ok=True)

    total_available = len(ds) if ds is not None else int(test_images.shape[0])
    num_to_export = min(5000, total_available)
    print(f"[INFO] Saving {num_to_export} MNIST test images to {out_dir}")

    for i in range(num_to_export):
        if ds is not None:
            img, _ = ds[i]
        else:
            arr = test_images[i, 0].numpy()
            arr = ((arr + 1.0) * 0.5 * 255.0).clip(0.0, 255.0).astype("uint8")
            img = Image.fromarray(arr, mode="L")
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
