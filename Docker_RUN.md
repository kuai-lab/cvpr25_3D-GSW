# 3D-GSW Docker Image: Loading and Running

The Docker image already contains the **code, Python environment, and watermark checkpoints**, so **no build step is required**.

## 0. Requirements

- x86_64 (amd64) Linux host
- NVIDIA driver
- NVIDIA Container Toolkit configured so that `docker run --gpus all ...` works
- Disk space:
  - Approximately 45 GB for the loaded Docker image
  - Approximately 31 GB for this directory
  - Additional space for outputs

## 1. Choose the Image

You only need **one** of the following images, depending on your GPU.

| File | Image Name | Supported GPUs | NVIDIA Driver |
|---|---|---|---|
| `image/3dgsw_s42-m19701105.tar` | `3dgsw:s42-m19701105` | sm_70–90 (V100, A100, A6000, L40, RTX 6000 Ada, H100, etc.) | 530 or later |
| `image/3dgsw_s42-m19701105-sm120.tar` | `3dgsw:s42-m19701105-sm120` | sm_70–120 (all GPUs above, plus B200, RTX PRO 6000 Blackwell, RTX 50 series) | 570 or later |

You can check your NVIDIA driver version at the top of the `nvidia-smi` output.

Both images produce the same results.

## 2. Load the Image

This only needs to be done once.

```bash
cd cvpr25_3dgsw_docker

# Optional: verify that the transferred archive is not corrupted.
# It is fine to have only the tar file corresponding to your GPU.
(cd image && sha256sum -c SHA256SUMS --ignore-missing)

docker load -i image/3dgsw_s42-m19701105.tar

# For Blackwell GPUs, use:
# docker load -i image/3dgsw_s42-m19701105-sm120.tar

# Verify that the image has been loaded.
docker images 3dgsw
```

## 3. Run

Run the following commands from this directory.

```bash
# Evaluation using the checkpoints included in the image,
# without training.
# This performs rendering, decoding, and PSNR/SSIM/LPIPS evaluation.

bash docker/run.sh infer32      # 32-bit, 25 scenes
bash docker/run.sh infer48      # 48-bit
bash docker/run.sh infer        # both 32-bit and 48-bit

# Training + evaluation.
# Both 32-bit and 48-bit settings are trained for 8 epochs.

bash docker/run.sh 32
bash docker/run.sh 48
bash docker/run.sh all          # 32-bit followed by 48-bit

# Select specific GPUs.
# By default, all visible GPUs are used.

GPU_IDS=0,1 bash docker/run.sh infer32
```

If you loaded the **sm120 image**, prepend the command with:

```bash
IMAGE=3dgsw:s42-m19701105-sm120 bash docker/run.sh infer32
```

Results are written to `output/`, and logs are written to `log/`.

After evaluation finishes, a result table is printed for each dataset.

`run.sh` mounts the following directories from this folder into the container:

- `dataset/` — mounted as read-only
- `gaussian_models/`
- `scripts/`

**Do not delete or move these directories.**

In particular, if `scripts/` is missing, Docker may mount an empty directory in its place, which will hide the scripts included in the image and prevent execution.

### VRAM Requirements

Evaluation requires relatively little VRAM.

During training, however, a single mip-NeRF 360 job may use up to approximately **30 GB of VRAM**, so a GPU with **32 GB or more VRAM** is recommended.

## 4. Training with a Different Seed

To train with different seeds, edit **only the seed values on line 5** of:

```text
scripts/run_s42_m19701105_32bit.sh
```

For the 48-bit setting, edit:

```text
scripts/run_s42_m19701105_48bit.sh
```

Original line:

```bash
BIT=32 SEED=42 MSG_SEED=19701105 EPOCH=8 EVAL_EPOCHS="8"
```

For example, to use model seed `0` and message seed `1234`:

```bash
BIT=32 SEED=0 MSG_SEED=1234 EPOCH=8 EVAL_EPOCHS="8"
```

Then run the same command as in Section 3:

```bash
bash docker/run.sh 32

# For 48-bit:
# bash docker/run.sh 48

# If using the sm120 image, prepend IMAGE=... as described above.
```

Because `scripts/` is mounted at runtime, you do **not** need to rebuild or reload the Docker image after modifying the seed.

## Directory Structure

```text
cvpr25_3dgsw_docker/
├── LOAD_AND_RUN.md       # This file
├── image/                # Docker image archive(s) + SHA256SUMS
├── docker/run.sh         # Container launcher
├── scripts/              # Runtime scripts, mounted into the container
├── dataset/              # mip-NeRF 360 (9), LLFF (8), NeRF Synthetic (8), 21 GB
└── gaussian_models/      # Pretrained 3DGS models, 9.6 GB
```