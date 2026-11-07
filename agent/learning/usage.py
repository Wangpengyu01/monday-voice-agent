"""What the teacher (DeepSeek) costs: one line per API call in run/learning/usage/<day>.jsonl.

Every call records the token counts DeepSeek returns (cache hit / cache miss input, output), which job
made it (school, practice, distill, the conversation ...) and whether it fell in the peak window, priced
with the owner's price table (2026-11, deepseek-flash, yuan per million tokens):

                     off-peak   peak
    input, cache hit   0.02     0.04
    input, cache miss  1        2
    output             4        8

Peak = Beijing time Mon-Fri 9:00-12:00 and 14:00-18:00 (Chinese public holidays are off-peak; they are
not known here, so a holiday counts as peak - the estimate errs on the expensive side).
Appends are one short line each (safe from several processes); a failure to write never breaks a call.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

USAGE_DIR = Path(os.environ.get("MONDAY_USAGE_DIR", "/userdata/agent/run/learning/usage"))
PRICES = {  # yuan per million tokens: (off-peak, peak)
    "hit": (0.02, 0.04),
    "miss": (1.0, 2.0),
    "out": (4.0, 8.0),
}
_BEIJING = dt.timezone(dt.timedelta(hours=8))
TAGS = {  # __main__ module -> job name shown on the page
    "learning.school": "上课", "learning.practice": "练习", "learning.distill": "蒸馏",
    "learning.reflect": "每晚反思", "learning.owner": "了解主人", "learning.inbox": "读文件",
    "orchestrator.main": "对话", "learning.lora_prep": "整理训练数据", "learning.selftune": "自我优化",
}


def is_peak(t: float | None = None) -> bool:
    now = dt.datetime.fromtimestamp(t or time.time(), _BEIJING)
    if now.weekday() >= 5:
        return False
    hm = now.hour * 60 + now.minute
    return 9 * 60 <= hm < 12 * 60 or 14 * 60 <= hm < 18 * 60


# Owner's rule (2026-11-06): in peak hours nothing but the live conversation may call the teacher -
# no lessons, practice, distillation or any other background job.
PEAK_ALLOWED = {"对话"}


def peak_blocked(t: float | None = None) -> bool:
    return is_peak(t) and job_tag() not in PEAK_ALLOWED


def job_tag() -> str:
    if os.environ.get("MONDAY_USAGE_TAG"):
        return os.environ["MONDAY_USAGE_TAG"]
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)
    name = getattr(spec, "name", "") or Path(getattr(main, "__file__", "") or "?").stem
    return TAGS.get(name, name)


def cost(hit: int, miss: int, out: int, peak: bool) -> float:
    i = 1 if peak else 0
    return (hit * PRICES["hit"][i] + miss * PRICES["miss"][i] + out * PRICES["out"][i]) / 1e6


def record(usage: dict | None, model: str = "") -> None:
    if not isinstance(usage, dict):
        return
    try:
        prompt = int(usage.get("prompt_tokens") or 0)
        hit = int(usage.get("prompt_cache_hit_tokens") or 0)
        miss = int(usage.get("prompt_cache_miss_tokens") or max(0, prompt - hit))
        out = int(usage.get("completion_tokens") or 0)
        t = time.time()
        peak = is_peak(t)
        line = {"t": round(t, 1), "job": job_tag(), "model": model, "hit": hit, "miss": miss, "out": out,
                "peak": peak, "yuan": round(cost(hit, miss, out, peak), 6)}
        USAGE_DIR.mkdir(parents=True, exist_ok=True)
        day = dt.datetime.fromtimestamp(t, _BEIJING).strftime("%Y-%m-%d")
        with (USAGE_DIR / f"{day}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except (OSError, ValueError, TypeError):
        pass


def summary(day: str, usage_dir: Path | None = None) -> dict:
    """Totals for one day: overall and per job."""
    f = (usage_dir or USAGE_DIR) / f"{day}.jsonl"
    total = {"calls": 0, "hit": 0, "miss": 0, "out": 0, "yuan": 0.0, "peak_yuan": 0.0}
    jobs: dict[str, dict] = {}
    if f.is_file():
        for raw in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(raw)
            except ValueError:
                continue
            j = jobs.setdefault(r.get("job") or "?", {"calls": 0, "hit": 0, "miss": 0, "out": 0, "yuan": 0.0})
            for d in (total, j):
                d["calls"] += 1
                for k in ("hit", "miss", "out"):
                    d[k] += int(r.get(k) or 0)
                d["yuan"] += float(r.get("yuan") or 0)
            if r.get("peak"):
                total["peak_yuan"] += float(r.get("yuan") or 0)
    total["yuan"] = round(total["yuan"], 3)
    total["peak_yuan"] = round(total["peak_yuan"], 3)
    for j in jobs.values():
        j["yuan"] = round(j["yuan"], 3)
    return {"day": day, "total": total, "jobs": dict(sorted(jobs.items(), key=lambda kv: -kv[1]["yuan"]))}
