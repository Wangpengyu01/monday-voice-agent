"""Push LoRA progress from the GPU server to the board's web page (进度 tab).

The board has no key for the GPU server, so this runs on the workstation that can reach both:

    python -I tools/lora/sync_progress.py --version v1 [--report path/to/report.md] [--switch-state ask|done|wait]

It reads train_<v>.log / export_<v>.log / out/lora_<v>/train_log.json on the GPU server, checks
whether the model is on the board, and writes run/learning/progress/lora.json there. Nothing secret
is read or sent: only logs and loss numbers.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

KEY = str(Path.home() / ".ssh" / "claude_board_ed25519")
GPU = "gpu@192.168.100.4"
BOARD = "youyeetoo@10.0.0.177"
BOARD_PROGRESS = "/userdata/agent/run/learning/progress"
BOARD_MODELS = "/userdata/models"


def ssh(host: str, cmd: str) -> str:
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-i", KEY, host, cmd],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    return r.stdout


def scp_to_board(local: Path, remote: str) -> None:
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-i", KEY, str(local), f"{BOARD}:{remote}"], check=True, timeout=60)


def build(version: str, switch_state: str) -> dict:
    sep = "\n===SEP===\n"
    raw = ssh(GPU, f"cd ~/monday-lora; tail -c 20000 train_{version}.log; echo '{sep}'; grep -aE '^==|MERGE_DONE|EXPORT_DONE|No space left|Error:' export_{version}.log | grep -v ModuleNotFound;"
                   f" echo '{sep}'; cat out/lora_{version}/train_log.json 2>/dev/null; echo '{sep}';"
                   f" pgrep -f train_lora.py >/dev/null && echo TRAINING; pgrep -f export_lora.sh >/dev/null && echo EXPORTING;"
                   f" ls out/monday_{version}/Qwen3.5-4B.rknn 2>/dev/null")
    parts = (raw.split("===SEP===") + ["", "", "", ""])[:4]
    train_log, export_log, tl, flags = parts
    on_board = ssh(BOARD, f"ls {BOARD_MODELS}/Qwen3.5-4B-monday-{version}/Qwen3.5-4B.rknn 2>/dev/null").strip()

    steps_m = re.findall(r"step (\d+)/(\d+) loss ([\d.]+)", train_log)
    train_done = "TRAIN_DONE" in train_log
    failed = re.search(r"Traceback|Error:|No space left", export_log)
    st = lambda done, running: "done" if done else ("running" if running else "wait")
    merge_done = "MERGE_DONE" in export_log
    onnx_done = "== export_rknn" in export_log
    rknn_done = "EXPORT_DONE" in export_log
    exporting = "EXPORTING" in flags
    steps = [
        {"name": "整理训练数据", "state": "done", "detail": "对话+课堂，去重、去掉含账号的样本"},
        {"name": "LoRA 训练（A100）", "state": st(train_done, "TRAINING" in flags),
         "detail": f"第 {steps_m[-1][0]}/{steps_m[-1][1]} 步 · loss {steps_m[-1][2]}" if steps_m else ""},
        {"name": "合并权重", "state": st(merge_done, exporting and not merge_done)},
        {"name": "量化 + 导出 ONNX", "state": st(onnx_done, exporting and merge_done and not onnx_done)},
        {"name": "编译成 RK1828 格式", "state": st(rknn_done, exporting and onnx_done and not rknn_done)},
        {"name": "传到板子", "state": "done" if on_board else "wait"},
        {"name": "备份后切换试听", "state": switch_state, "detail": "切换前先打快照，不好就回滚"},
    ]
    if failed and not rknn_done and not exporting:
        first = next(s for s in steps[2:5] if s["state"] != "done")   # the export step that broke
        first["state"], first["detail"] = "failed", failed.group(0)
    stage = next((f"正在：{s['name']}" for s in steps if s["state"] in ("running", "failed", "ask")), "")
    if not stage:
        stage = "全部完成" if all(s["state"] == "done" for s in steps) else "等待下一步：" + next(s["name"] for s in steps if s["state"] == "wait")

    version_info = {"version": version}
    try:
        log = json.loads(tl)
        version_info.update({
            "samples": None, "loss_before": round(log["before"]["eval_loss"], 3), "loss_after": round(log["after"]["eval_loss"], 3),
            "compare": [{"user": b["user"][:80], "before": b["got"][:140], "after": a["got"][:140]}
                        for b, a in zip(log["before"]["samples"], log["after"]["samples"])]})
    except (ValueError, KeyError):
        pass
    m = re.search(r"train (\d+) eval (\d+)", train_log)
    if m:
        version_info["samples"] = int(m.group(1)) + int(m.group(2))
    return {"updated": time.strftime("%Y-%m-%d %H:%M"), "stage": stage, "steps": steps, "versions": [version_info]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--report", help="also publish this markdown work report")
    ap.add_argument("--switch-state", default="wait", choices=["wait", "ask", "running", "done", "failed"])
    a = ap.parse_args()
    data = build(a.version, a.switch_state)
    ssh(BOARD, f"mkdir -p {BOARD_PROGRESS}/reports")
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "lora.json"
        f.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        scp_to_board(f, f"{BOARD_PROGRESS}/lora.json")
    if a.report:
        scp_to_board(Path(a.report), f"{BOARD_PROGRESS}/reports/{Path(a.report).name}")
    print(data["stage"])
    for s in data["steps"]:
        print(f"  [{s['state']}] {s['name']} {s.get('detail', '')}")


if __name__ == "__main__":
    main()
