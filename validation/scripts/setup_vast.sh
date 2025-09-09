#!/usr/bin/env bash
#
# setup_vast.sh — Prepare a Vast.ai GPU instance for training.
#
# WHAT THIS DOES
#   • Installs tmux, git, rsync (if missing)
#   • Installs Docker (if missing) and NVIDIA container toolkit
#   • Creates /workspace/pae (code + data mount point)
#   • Provides a helper to launch a CUDA‑enabled PyTorch container
#
# PREREQS
#   • A running Vast.ai GPU VM (choose an image with Ubuntu 22.04+)
#   • SSH access (you can ProxyJump via your Hetzner VPS if desired)
#
# USAGE
#   bash setup_vast.sh     #     --workspace /workspace/pae     #     --image pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime     #     --shm 16g
#
# AFTER RUNNING
#   Enter the container:
#     docker run --gpus all --rm -it --shm-size=16g -v /workspace/pae:/workspace/pae pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime bash
#   Then set up venv, install requirements, and launch your sweeps.
#
set -euo pipefail

# ---------- defaults ----------
WORKDIR="${WORKDIR:-/workspace/pae}"
IMAGE="${IMAGE:-pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime}"
SHM_SIZE="${SHM_SIZE:-16g}"

# ---------- arg parsing ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) WORKDIR="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --shm) SHM_SIZE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

echo "[setup_vast] Using workspace: $WORKDIR"
sudo mkdir -p "$WORKDIR"
sudo chown -R "$USER":"$USER" "$WORKDIR"

echo "[setup_vast] Installing basics..."
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y tmux git rsync curl ca-certificates

# Docker install if missing
if ! command -v docker >/dev/null 2>&1; then
  echo "[setup_vast] Installing Docker..."
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER" || true
else
  echo "[setup_vast] Docker present."
fi

# NVIDIA container toolkit (for --gpus all)
if ! dpkg -l | grep -q nvidia-container-toolkit; then
  echo "[setup_vast] Installing NVIDIA container toolkit..."
  distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -y && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure
  sudo systemctl restart docker
else
  echo "[setup_vast] NVIDIA container toolkit present."
fi

echo "[setup_vast] GPU visibility check:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "  NOTE: nvidia-smi not found on host. It will be available inside CUDA containers."
fi

cat <<EOF

[setup_vast] Done.

Workspace: $WORKDIR
Default image: $IMAGE
Shared memory: $SHM_SIZE

To start a CUDA‑enabled PyTorch container and mount your workspace:

  docker run --gpus all --rm -it \        --shm-size=$SHM_SIZE \        -v $WORKDIR:/workspace/pae \        $IMAGE bash

Inside the container, you might run:

  cd /workspace/pae
  python -m venv .venv && . .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  # then your training commands

EOF
