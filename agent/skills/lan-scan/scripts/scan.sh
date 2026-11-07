#!/bin/bash
# Read-only LAN scan: ping sweep of a /24 (or one host), then the ARP table. Usage:
#   scan.sh                 the board's own /24
#   scan.sh 10.0.0.15       one address
#   scan.sh 192.168.100     another /24 prefix
set -u
arg="${1:-}"
if [[ "$arg" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
  targets=("$arg")
else
  if [[ "$arg" =~ ^([0-9]{1,3}\.){2}[0-9]{1,3}$ ]]; then
    prefix="$arg"
  else
    src=$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i=="src") print $(i+1)}' | head -1)
    prefix="${src%.*}"
  fi
  [ -n "$prefix" ] || { echo "找不到本机网段"; exit 1; }
  targets=()
  for i in $(seq 1 254); do targets+=("$prefix.$i"); done
fi

tmp=$(mktemp -d)
for ip in "${targets[@]}"; do
  ( ping -c 1 -W 1 "$ip" >/dev/null 2>&1 && touch "$tmp/$ip" ) &
  # keep at most ~64 pings in flight
  while [ "$(jobs -rp | wc -l)" -ge 64 ]; do sleep 0.05; done
done
wait

printf "%-16s %-18s %-10s %s\n" IP MAC 状态 主机名
count=0
for ip in "${targets[@]}"; do
  mac=$(ip neigh show "$ip" 2>/dev/null | awk '/lladdr/ {print $5; exit}')
  if [ -e "$tmp/$ip" ]; then state="在线"; elif [ -n "$mac" ]; then state="不回ping"; else continue; fi
  name=$(getent hosts "$ip" 2>/dev/null | awk '{print $2; exit}')
  printf "%-16s %-18s %-10s %s\n" "$ip" "${mac:--}" "$state" "${name:-}"
  count=$((count + 1))
done
rm -rf "$tmp"
echo "共 $count 台"
