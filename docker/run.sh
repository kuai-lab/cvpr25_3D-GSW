# Run the model-seed-42 / message-seed-19701105 3D-GSW experiments in the image.
#   bash docker/run.sh 32              # train + eval 32bit: 8 epochs
#   bash docker/run.sh 48              # train + eval 48bit: 8 epochs
#   bash docker/run.sh all             # 32bit then 48bit
#   bash docker/run.sh infer32         # no training: evaluate the packaged 32bit epoch-8 checkpoints
#   bash docker/run.sh infer48         # no training: evaluate the packaged 48bit epoch-8 checkpoints
#   bash docker/run.sh infer           # both
#   GPU_IDS=4,5,6,7 bash docker/run.sh 32   # host GPUs to use (default: all)
# Data comes from this directory: dataset/ (read-only), gaussian_models/ (pretrained reference renders are written here),
# results go to output/ and log/. scripts/ is mounted from here too, so editing a run script (e.g. MSG_SEED) needs no rebuild.
set -e
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=${IMAGE:-3dgsw:s42-m19701105}
case ${1:-} in
    32|48)   CMD="bash scripts/run_s42_m19701105_${1}bit.sh" ;;
    all)     CMD="bash scripts/run_s42_m19701105.sh" ;;
    infer32) CMD="bash scripts/infer_checkpoints.sh 32bit_seed42_msg19701105_ep8" ;;
    infer48) CMD="bash scripts/infer_checkpoints.sh 48bit_seed42_msg19701105_ep8" ;;
    infer)   CMD="bash scripts/infer_checkpoints.sh 32bit_seed42_msg19701105_ep8 48bit_seed42_msg19701105_ep8" ;;
    *)       echo "usage: bash docker/run.sh <32|48|all|infer32|infer48|infer>"; exit 1 ;;
esac
GPU_ARG=all; [ -n "${GPU_IDS:-}" ] && GPU_ARG="\"device=$GPU_IDS\""
mkdir -p "$ROOT/output" "$ROOT/log"
docker run --rm --init --gpus "$GPU_ARG" --shm-size 16g --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$ROOT/dataset":/opt/3dgsw/dataset:ro \
    -v "$ROOT/gaussian_models":/opt/3dgsw/gaussian_models \
    -v "$ROOT/output":/opt/3dgsw/output \
    -v "$ROOT/log":/opt/3dgsw/log \
    -v "$ROOT/scripts":/opt/3dgsw/scripts:ro \
    "$IMAGE" $CMD
