"""Merge a LoRA adapter into Qwen3.5-4B and save a full HF checkpoint the rknn3 export scripts can read.

    python merge_lora.py --base models/Qwen3.5-4B --adapter out/lora_v1 --out models/Qwen3.5-4B-monday-v1
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForImageTextToText


def keys(d: Path) -> set[str]:
    idx = d / "model.safetensors.index.json"
    if idx.is_file():
        return set(json.loads(idx.read_text())["weight_map"])
    out: set[str] = set()
    for f in d.glob("*.safetensors"):
        with safe_open(str(f), "pt") as h:
            out |= set(h.keys())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base, out = Path(a.base), Path(a.out)

    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.save_pretrained(out, max_shard_size="5GB")
    for f in base.iterdir():   # tokenizer, chat template, processor configs
        if f.is_file() and not f.name.startswith("model.safetensors") and f.name != "config.json":
            shutil.copy2(f, out / f.name)

    # mtp.* (multi-token prediction head) is in the checkpoint but never loaded by the HF model nor used
    # by the rknn3 export, so it is not carried over
    kb, ko = {k for k in keys(base) if not k.startswith("mtp.")}, keys(out)
    if kb != ko:
        raise SystemExit(f"weight names differ: missing {sorted(kb - ko)[:5]} extra {sorted(ko - kb)[:5]}")
    print(f"merged {len(ko)} tensors -> {out}")
    print("MERGE_DONE")


if __name__ == "__main__":
    main()
