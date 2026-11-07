#!/bin/bash
# Merge LoRA -> HF checkpoint -> ONNX (GRQ W4A16) -> RKNN for RK1820/1828, on the GPU server.
# Lives on /data (150G disk). The merged checkpoint is deleted once exported (re-made from base + adapter).
# usage: export_lora.sh v1
set -e
V=${1:?version, e.g. v1}
ROOT=~/monday-lora
PY=$ROOT/q35venv/bin/python
BASE=$ROOT/models/Qwen3.5-4B
MERGED=$ROOT/models/Qwen3.5-4B-monday-$V
OUT=$ROOT/out/monday_$V
mkdir -p $OUT
echo "== merge $(date)"; df -h ~ | tail -1
$PY $ROOT/merge_lora.py --base $BASE --adapter $ROOT/out/lora_$V --out $MERGED
# (base weights are kept now that /data has room)
cd $ROOT/zoo110/rknn3-model-zoo/examples/Qwen3_5/python
echo "== export_llm $(date)"; df -h ~ | tail -1
$PY export_llm.py --model_path $MERGED --quant \
    --cali_dataset $ROOT/zoo110/rknn3-model-zoo/datasets/llm_quant.json --export_llm_path $OUT/Qwen3.5-4B.onnx
rm -rf $MERGED
echo "== export_rknn $(date)"; df -h ~ | tail -1
$PY export_rknn.py --onnx_path $OUT/Qwen3.5-4B.onnx --config $OUT/Qwen3.5-4B.config.pkl \
    --rknn_path $OUT/Qwen3.5-4B.rknn --platform rk1820
rm -rf tmp/const_cache tmp/check_external_data tmp/embed.npy tmp/model_build tmp/check*_*.onnx
echo "== done $(date)"; df -h ~ | tail -1
ls -la $OUT | grep -v onnx__
echo EXPORT_DONE
