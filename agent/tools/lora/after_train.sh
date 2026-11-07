#!/bin/bash
# On the GPU server, by itself (no PC needed): wait for train_lora.py of version V to finish, then run
# export_lora.sh V with a temporary swap file. v4's export_rknn was OOM-killed twice at "register
# configuration" on the 29 GB machine and only finished with swap (2026-11-06).
#   cd ~/monday-lora && nohup bash after_train.sh v5 > export_v5.log 2>&1 < /dev/null &
# The log stays empty until training ends (ship_to_board.sh relies on that).
V=${1:?version, e.g. v5}
cd ~/monday-lora || exit 1
while pgrep -f "train_lora.py --data train_$V.jsonl" >/dev/null; do sleep 60; done
grep -qa TRAIN_DONE train_$V.log || { echo "TRAIN_FAILED $(date)"; tail -c 1500 train_$V.log; exit 1; }
SWAP=/swap_${V}export
cleanup() { sudo -n /sbin/swapoff $SWAP 2>/dev/null; sudo -n rm -f $SWAP; echo "== swap removed $(date)"; }
trap cleanup EXIT
sudo -n fallocate -l 32G $SWAP && sudo -n chmod 600 $SWAP && sudo -n /sbin/mkswap $SWAP >/dev/null \
  && sudo -n /sbin/swapon $SWAP && echo "== swap on ($SWAP, 32G) $(date)" || echo "== swap could not be added, trying without"
bash export_lora.sh $V
