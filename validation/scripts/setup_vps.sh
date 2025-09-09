#!/usr/bin/env bash
#
# setup_vps.sh — Bootstrap a Hetzner Cloud VPS as your always‑on tmux/orchestration host.
#
# WHAT THIS DOES
#   • Installs tmux, git, rsync, Python venv, rclone, htop, unzip
#   • (Optional) Installs Docker (for light CPU eval or building images)
#   • Enables basic firewall (UFW) and fail2ban
#   • Creates a workspace with standard folders (data, logs, checkpoints)
#   • (Optional) Creates a swapfile (default 4G) for small VPS plans
#
# PREREQS
#   • Ubuntu 22.04/24.04 on a Hetzner CPX/CCX VPS (no GPU)
#   • Run as a sudo‑capable user:  bash setup_vps.sh [--workspace ~/work/pae]
#
# USAGE
#   bash setup_vps.sh     #     --workspace ~/work/pae     #     --swap 4G     #     --with-docker
#
# NEXT STEPS (after this finishes)
#   1) Create SSH config on your laptop with aliases 'vps' (this host) and 'vast' (your Vast.ai box).
#   2) Clone your repo into $WORKDIR and use tmux to drive runs.
#
set -euo pipefail

# ---------- defaults ----------
WORKDIR="${WORKDIR:-$HOME/work/pae}"
SWAP_SIZE="0"          # e.g., 4G to enable swap; 0 to skip
WITH_DOCKER="0"        # 1 to install docker

# ---------- arg parsing ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) WORKDIR="$2"; shift 2 ;;
    --swap) SWAP_SIZE="$2"; shift 2 ;;
    --with-docker) WITH_DOCKER="1"; shift 1 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

echo "[setup_vps] Using workspace: $WORKDIR"
mkdir -p "$WORKDIR"/{data,logs,checkpoints,reports}

echo "[setup_vps] Updating packages..."
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y       tmux git rsync python3-venv rclone htop unzip       ufw fail2ban

echo "[setup_vps] Hardening basic services (UFW + fail2ban)..."
sudo ufw allow OpenSSH
sudo ufw --force enable || true
sudo systemctl enable --now fail2ban

if [[ "$SWAP_SIZE" != "0" ]]; then
  echo "[setup_vps] Creating swapfile ($SWAP_SIZE)..."
  SWAPFILE="/swapfile"
  if [[ ! -f "$SWAPFILE" ]]; then
    sudo fallocate -l "$SWAP_SIZE" "$SWAPFILE" || sudo dd if=/dev/zero of="$SWAPFILE" bs=1M count=$(echo "$SWAP_SIZE" | sed 's/G//')000
    sudo chmod 600 "$SWAPFILE"
    sudo mkswap "$SWAPFILE"
    echo "$SWAPFILE none swap sw 0 0" | sudo tee -a /etc/fstab >/dev/null
  fi
  sudo swapon -a
  echo "[setup_vps] Swap active:"
  swapon --show || true
else
  echo "[setup_vps] Skipping swap setup."
fi

if [[ "$WITH_DOCKER" == "1" ]]; then
  echo "[setup_vps] Installing Docker..."
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER" || true
  echo "[setup_vps] Docker installed. You may need to re-login for group changes to apply."
else
  echo "[setup_vps] Skipping Docker install."
fi

cat <<EOF

[setup_vps] Done.

Workspace: $WORKDIR
Folders:   data/ logs/ checkpoints/ reports/

Helpful next steps:
  # create a tmux session
  tmux new -s pae

  # (optional) add SSH config on your laptop (~/.ssh/config)
  # Host vps
  #   HostName <hetzner-ip>
  #   User <user>
  #   IdentityFile ~/.ssh/id_ed25519
  #
  # Host vast
  #   HostName <vast-host-or-ip>
  #   Port <vast-ssh-port>
  #   User <vast-user>
  #   ProxyJump vps
  #   IdentityFile ~/.ssh/id_ed25519

EOF
