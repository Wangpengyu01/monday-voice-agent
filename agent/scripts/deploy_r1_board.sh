#!/bin/bash
# 从本机（Git Bash）把当前提交部署到 R1 板子。
#
#   bash agent/scripts/deploy_r1_board.sh            # 部署 HEAD
#   bash agent/scripts/deploy_r1_board.sh <ref>      # 部署指定分支/提交
#
# 用 git archive 打包；必须关掉 core.autocrlf，否则 Windows 上打出来的脚本是 CRLF，板上 bash 跑不了。
# 不要在这块板子上用 install_autostart.sh，原因见 board_apply_deploy.sh。
set -euo pipefail

REF="${1:-HEAD}"
BOARD="${R1_BOARD:-youyeetoo@10.0.0.177}"
KEY="${R1_KEY:-$HOME/.ssh/claude_board_ed25519}"
SSH=(ssh -i "$KEY" -o IdentitiesOnly=yes -o ConnectTimeout=8 "$BOARD")

cd "$(git rev-parse --show-toplevel)"
if [ "$REF" = "HEAD" ] && [ -n "$(git status --porcelain -- agent)" ]; then
  echo "agent/ 有未提交的改动，先提交再部署（部署的是提交里的内容）" >&2
  exit 1
fi
COMMIT=$(git rev-parse --short "$REF")
echo "部署 $REF ($COMMIT) -> $BOARD"

"${SSH[@]}" "rm -rf /tmp/r1-deploy && mkdir -p /tmp/r1-deploy"
git -c core.autocrlf=false archive "$REF" agent | "${SSH[@]}" "tar xf - -C /tmp/r1-deploy && echo $COMMIT > /tmp/r1-deploy/agent/DEPLOYED_COMMIT"
"${SSH[@]}" "sudo bash /tmp/r1-deploy/agent/scripts/board_apply_deploy.sh /tmp/r1-deploy/agent"
