#!/usr/bin/env bash
#
# ops.sh — Minimal helper for syncing and tunnels between VPS and Vast.ai
#
# OVERVIEW
#   Run this from your Hetzner VPS (for rsync) or from your laptop (for TB tunnel).
#
# PREREQS
#   • SSH aliases set in ~/.ssh/config:
#       Host vps
#         HostName <hetzner-ip>
#         User <user>
#         IdentityFile ~/.ssh/id_ed25519
#       Host vast
#         HostName <vast-ip-or-host>
#         Port <vast-ssh-port>
#         User <vast-user>
#         ProxyJump vps
#         IdentityFile ~/.ssh/id_ed25519
#   • rsync installed on both ends.
#
# USAGE
#   ./ops.sh to   <local_dir>  <remote_dir>         # Run on VPS → push to Vast
#   ./ops.sh from <remote_dir> <local_dir>          # Run on VPS → pull from Vast
#   ./ops.sh tb [local_port] [remote_port]          # Run on laptop → tunnel TB via ProxyJump (default 16006 6006)
#
# EXAMPLES
#   ./ops.sh to   ~/work/pae/  /workspace/pae/
#   ./ops.sh from /workspace/pae/logs/  ~/work/pae/logs/
#   ./ops.sh tb   16006 6006   # then open http://localhost:16006
#
set -euo pipefail

usage() {
  sed -n '1,80p' "$0" | sed 's/^\s\{4\}//'
}

cmd="${1:-}"
case "$cmd" in
  to)
    # Push local -> Vast
    # args: <local_dir> <remote_dir>
    src="${2:-}"; dst="${3:-}"
    if [[ -z "${src}" || -z "${dst}" ]]; then usage; exit 1; fi
    rsync -avz --delete "${src%/}/" vast:"${dst%/}/"
    ;;
  from)
    # Pull Vast -> local
    # args: <remote_dir> <local_dir>
    src="${2:-}"; dst="${3:-}"
    if [[ -z "${src}" || -z "${dst}" ]]; then usage; exit 1; fi
    rsync -avz vast:"${src%/}/" "${dst%/}/"
    ;;
  tb)
    # Launch TensorBoard SSH tunnel (run from your laptop)
    # args: [local_port] [remote_port]
    lport="${2:-16006}"
    rport="${3:-6006}"
    echo "Run this on your LAPTOP (requires ssh config with ProxyJump vps):"
    echo ""
    echo "  ssh -N -L ${lport}:localhost:${rport} vast"
    echo ""
    echo "Then open: http://localhost:${lport}"
    ;;
  *)
    usage
    exit 1
    ;;
esac
