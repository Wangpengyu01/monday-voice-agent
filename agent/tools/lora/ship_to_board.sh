#!/bin/bash
# Wait for export_lora.sh on the GPU server, then relay the model to the board through this PC
# (the board has no key for the GPU server).  usage: ship_to_board.sh v1 [staging dir]
# Only copies into a NEW directory on the board; the running model is not touched.
set -e
V=${1:?version}
STAGE=${2:-/f/monday_models}/$V
K=$HOME/.ssh/claude_board_ed25519
GPU=gpu@192.168.100.4
BOARD=youyeetoo@10.0.0.177
DST=/userdata/models/Qwen3.5-4B-monday-$V
HERE=$(cd "$(dirname "$0")" && pwd)
g() { ssh -o BatchMode=yes -i $K $GPU "$@"; }

# done = EXPORT_DONE, or the export script is gone (failed). Tracebacks alone don't count: the stock
# tokenizer-to-gguf helper always fails under the system python and the export carries on without it.
# (the export log only appears once training has finished, so "not running" counts only after that)
until g "grep -qa EXPORT_DONE ~/monday-lora/export_$V.log 2>/dev/null || { test -s ~/monday-lora/export_$V.log && ! ps -eo args | grep -q '^bash export_lora[.]sh'; } || grep -qa Traceback ~/monday-lora/train_$V.log"; do
  python -I "$HERE/sync_progress.py" --version "$V" >/dev/null 2>&1 || true
  sleep 120
done
python -I "$HERE/sync_progress.py" --version "$V" || true
g "grep -qa EXPORT_DONE ~/monday-lora/export_$V.log" || { echo "EXPORT_FAILED"; g "tail -c 1500 ~/monday-lora/export_$V.log"; exit 1; }

mkdir -p "$STAGE"
for f in Qwen3.5-4B.rknn Qwen3.5-4B.weight Qwen3.5-4B.embed.bin; do
  scp -q -o BatchMode=yes -i $K "$GPU:monday-lora/out/monday_$V/$f" "$STAGE/"
done
ssh -o BatchMode=yes -i $K $BOARD "mkdir -p $DST && cp /userdata/models/Qwen3.5-4B/Qwen3.5-4B.tokenizer.gguf $DST/"
scp -q -o BatchMode=yes -i $K "$STAGE"/Qwen3.5-4B.* "$BOARD:$DST/"
( cd "$STAGE" && sha256sum Qwen3.5-4B.* | sed "s/ [*]/  /" ) > "$STAGE/sha256.txt"
ssh -o BatchMode=yes -i $K $BOARD "cd $DST && sha256sum Qwen3.5-4B.rknn Qwen3.5-4B.weight Qwen3.5-4B.embed.bin" > "$STAGE/sha256.board.txt"
grep -vF tokenizer "$STAGE/sha256.txt" | sort | diff - <(sort "$STAGE/sha256.board.txt") && echo "CHECKSUM_OK"
ssh -o BatchMode=yes -i $K $BOARD "ls -la $DST"
python -I "$HERE/sync_progress.py" --version "$V" --switch-state ask
echo SHIP_DONE
