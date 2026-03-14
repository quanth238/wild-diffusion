# Few-shot Transfer Reproduction Guide (Section 4.3, Table 2/6)

Tài liệu này hướng dẫn chạy **few-shot transfer** cho WILD-Diffusion trong môi trường server của bạn:
- Conda env: `quanth`
- Data root: `/mnt/data/quanth`
- Project: `/home/quanth/working_space/Wild-Diffusion`

Mục tiêu:
- Chạy được setting **pretrained backbone + fine-tune/adapt** trên bộ 100-shot.
- Đánh giá đúng protocol few-shot của paper: **FID với 5k mẫu**, reference là tập train target.

## 1) Những gì paper nói rõ (để bám đúng)

Từ Section 4.3 và Appendix C.4:
- Few-shot datasets: `Obama`, `Grumpy Cat`, `Panda` (100-shot) và `AnimalFace`.
- So sánh pretrained và non-pretrained.
- FID few-shot tính với **5,000 generated samples**.
- Với các phương pháp transfer, backbone pretrained lấy từ **FFHQ**.

Các hyperparameter được paper nêu rõ cho WILD-Diffusion:
- `m = 20`
- `K = 5`
- `eta = 0.01`
- `gamma = 1`
- warmup ratio dùng 20% (`Sw/S = 0.2`)

## 2) Giới hạn tái lập trong repo này (quan trọng)

- Repo này có pipeline diffusion (`train.py`, `--transfer`) nên có thể chạy few-shot transfer theo hướng **diffusion**.
- **Table 6 (GAN architecture / StyleGAN-v2)** không có code tương ứng trong repo này, nên không thể tái lập end-to-end chỉ bằng repo hiện tại.
- Paper không public đầy đủ toàn bộ hyperparameter riêng cho từng run few-shot; vì vậy kết quả số tuyệt đối có thể lệch, dù protocol và thiết lập cốt lõi có thể bám sát.

## 3) Chuẩn bị thư mục

```bash
mkdir -p /mnt/data/quanth/{datasets/fewshot,checkpoints/wild-diffusion,experiments/wild-diffusion/fewshot}
```

## 4) Tải pretrained backbone (FFHQ)

Nguồn: pretrained EDM từ NVIDIA (FFHQ 64x64, NCSN++).

```bash
cd /mnt/data/quanth/checkpoints/wild-diffusion
curl -L --retry 5 --retry-delay 3 \
  -o edm-ffhq-64x64-uncond-vp.pkl \
  https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl
```

Nếu cluster chặn internet ở worker node:
- download từ máy local rồi `scp/rsync` lên `/mnt/data/quanth/checkpoints/wild-diffusion/`.

## 5) Tải few-shot datasets (100-shot)

Các bộ 100-shot thường dùng theo DiffAugment benchmark:

```bash
cd /mnt/data/quanth/datasets/fewshot
curl -L -O https://hanlab.mit.edu/projects/data-efficient-gans/datasets/100-shot-obama.zip
curl -L -O https://hanlab.mit.edu/projects/data-efficient-gans/datasets/100-shot-grumpy_cat.zip
curl -L -O https://hanlab.mit.edu/projects/data-efficient-gans/datasets/100-shot-panda.zip

unzip -o 100-shot-obama.zip -d raw/100-shot-obama
unzip -o 100-shot-grumpy_cat.zip -d raw/100-shot-grumpy_cat
unzip -o 100-shot-panda.zip -d raw/100-shot-panda
```

`AnimalFace` không có downloader chuẩn ngay trong repo này; nếu bạn có dữ liệu nguồn, chỉ cần đặt vào:
- `/mnt/data/quanth/datasets/fewshot/raw/animalface-cat`
- `/mnt/data/quanth/datasets/fewshot/raw/animalface-dog`

## 6) Chọn profile tái lập: practical (64) vs paper-like (256)

Paper ghi few-shot datasets ở 256x256. Tuy nhiên checkpoint pretrained public dễ dùng nhất là FFHQ **64x64**.

- `practical_transfer_64` (khuyến nghị để chạy ổn định ngay):
  - dùng checkpoint `edm-ffhq-64x64-uncond-vp.pkl`
  - resize target dataset về 64x64
- `paper_like_256` (gần độ phân giải paper hơn):
  - cần source checkpoint FFHQ 256x256 tương thích kiến trúc (repo hiện không cung cấp sẵn)
  - hoặc bạn tự pretrain FFHQ 256 trước rồi mới fine-tune few-shot 256

Phần bên dưới dùng profile `practical_transfer_64`.

### 6.1) Chuẩn hóa dữ liệu target về 64x64 để khớp pretrained FFHQ64

Checkpoint FFHQ pretrained ở trên là 64x64, nên để transfer đúng kiến trúc, hãy resize/crop target về 64x64:

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SIZE = 64

def convert(src_dir, dst_dir):
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    idx = 0
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        img = Image.open(p).convert("RGB")
        img = ImageOps.fit(
            img, (SIZE, SIZE),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        img.save(dst / f"{idx:05d}.png")
        idx += 1
    print(f"{src} -> {dst}: {idx} images")

convert("/mnt/data/quanth/datasets/fewshot/raw/100-shot-obama", "/mnt/data/quanth/datasets/fewshot/100-shot-obama-64")
convert("/mnt/data/quanth/datasets/fewshot/raw/100-shot-grumpy_cat", "/mnt/data/quanth/datasets/fewshot/100-shot-grumpy_cat-64")
convert("/mnt/data/quanth/datasets/fewshot/raw/100-shot-panda", "/mnt/data/quanth/datasets/fewshot/100-shot-panda-64")
PY
```

## 7) Submit job few-shot pretrained transfer (SLURM)

Ví dụ chạy cho `Obama`:

```bash
cd /home/quanth/working_space

PROJECT_DIR=/home/quanth/working_space/Wild-Diffusion \
./scripts/slurm_workflow.sh submit wd-fs-obama-pretrained -- \
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --standalone --nproc_per_node=1 train.py \
    --outdir=/mnt/data/quanth/experiments/wild-diffusion/fewshot \
    --data=/mnt/data/quanth/datasets/fewshot/100-shot-obama-64 \
    --cond=0 \
    --arch=ncsnpp \
    --precond=wdroedm \
    --transfer=/mnt/data/quanth/checkpoints/wild-diffusion/edm-ffhq-64x64-uncond-vp.pkl \
    --duration=20 \
    --batch=64 \
    --batch-gpu=32 \
    --lr=2e-4 \
    --workers=8 \
    --augment=0.15 \
    --fp16=1 \
    --cres=1,2,2,2 \
    --wdro-warmup-ratio=0.2 \
    --wdro-m-epochs=20 \
    --wdro-k=5 \
    --wdro-step-size=0.01 \
    --wdro-gamma=1.0 \
    --wdro-p-adv=1.0
```

Gợi ý chạy non-pretrained (để so với pretrained):
- giữ nguyên lệnh trên, chỉ **bỏ** `--transfer=...`.

## 8) Đánh giá FID theo protocol few-shot (5k samples)

Sau khi train xong, lấy snapshot cuối:

```bash
RUN_DIR=$(ls -1dt /mnt/data/quanth/experiments/wild-diffusion/fewshot/* | head -n1)
SNAP=$(ls -1 "${RUN_DIR}"/network-snapshot-*.pkl | tail -n1)
echo "$RUN_DIR"
echo "$SNAP"
```

Sinh 5k samples:

```bash
cd /home/quanth/working_space/Wild-Diffusion
torchrun --standalone --nproc_per_node=1 generate.py \
  --network="${SNAP}" \
  --outdir="${RUN_DIR}/gen-5k" \
  --seeds=0-4999 \
  --batch=64
```

Tạo reference stats từ tập train target:

```bash
python fid.py ref \
  --data=/mnt/data/quanth/datasets/fewshot/100-shot-obama-64 \
  --dest="${RUN_DIR}/ref-obama-64.npz" \
  --batch=64
```

Tính FID (5k):

```bash
torchrun --standalone --nproc_per_node=1 fid.py calc \
  --images="${RUN_DIR}/gen-5k" \
  --ref="${RUN_DIR}/ref-obama-64.npz" \
  --num=5000 \
  --batch=64
```

## 9) Lặp cho dataset khác

Chỉ cần thay:
- `--data=/mnt/data/quanth/datasets/fewshot/<dataset>-64`
- tên job `wd-fs-...`
- file ref output.

Ví dụ:
- `100-shot-grumpy_cat-64`
- `100-shot-panda-64`
- `animalface-cat-64` / `animalface-dog-64` (nếu bạn đã chuẩn bị dữ liệu nguồn).

## 10) Lưu ý để “gần paper” nhất

- Dùng đúng WDRO config: `warmup=0.2, m=20, K=5, step=0.01, gamma=1`.
- So sánh cả 2 setting: pretrained vs non-pretrained.
- Dùng FID 5k và reference là training set của target domain.
- Chạy nhiều seed và báo cáo mean/std để giảm dao động few-shot.

## 11) Nguồn tham chiếu

- WILD-Diffusion paper (Section 4.3, Table 2/6): `pdfs/9149_WILD_Diffusion_A_WDRO_Ins.pdf`
- Official EDM pretrained checkpoint list (NVIDIA): https://github.com/NVlabs/edm
- 100-shot benchmark datasets (DiffAugment repo links): https://github.com/mit-han-lab/data-efficient-gans
