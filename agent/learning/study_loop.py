"""星期一 never sits idle: lesson -> practice round -> lesson -> practice round ... while the owner is quiet.

r1-study.service runs this loop (it replaces the r1-school / r1-practice timers):
  - before every run: has the owner said a real sentence in the last IDLE_MIN minutes
    (run/last_activity, see orchestrator _mark_activity)? then wait and check again;
  - a lesson = `python3 -m learning.school`, a practice round = `python3 -m learning.practice --slot N`
    (N counts up, so the four practice modes - knowledge / speaking / coding / agent - take turns);
  - each run holds the shared study lock (/tmp/monday-study.lock) like before, and the loop pauses a
    moment between runs so the nightly reflection (which waits on that lock) can get in;
  - a run that fails is logged and the loop moves on after a short pause; nothing here is fatal;
  - DeepSeek peak hours (learning.usage.is_peak: weekdays 9-12, 14-18): lessons and practice both ask the
    teacher, and the owner's rule is no teacher calls in peak hours, so the loop waits them out.
The page reads run/learning/study_loop.json (what comes next, whether it is waiting for the owner).

    cd /userdata/agent && python3 -m learning.study_loop
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from learning.practice import IDLE_MIN, last_activity
from learning.usage import is_peak

LOG = logging.getLogger("study_loop")
AGENT_ROOT = Path(__file__).resolve().parents[1]
RUN = AGENT_ROOT / "run"
STATE = RUN / "learning" / "study_loop.json"
LOCK = "/tmp/monday-study.lock"
GAP_S = 5            # between runs: lets a waiting reflect/manual run take the lock
PEAK_POLL_S = 60     # while DeepSeek is in its peak hours
OWNER_POLL_S = 30    # while the owner is talking
FAIL_PAUSE_S = 60
TIMEOUT = {"lesson": 1800, "practice": 900}


def load_state() -> dict:
    try:
        d = json.loads(STATE.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(d: dict) -> None:
    try:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        tmp.replace(STATE)
    except OSError:
        pass


def owner_quiet() -> bool:
    return time.time() - last_activity(RUN) > IDLE_MIN * 60


def command(kind: str, slot: int) -> list[str]:
    module = ["-m", "learning.school"] if kind == "lesson" else ["-m", "learning.practice", "--slot", str(slot)]
    return ["/usr/bin/flock", "-w", "1800", LOCK, sys.executable, *module]


def run_one(kind: str, slot: int) -> int:
    log = AGENT_ROOT / "logs" / ("school.log" if kind == "lesson" else "practice.log")
    with log.open("a", encoding="utf-8") as out:
        try:
            return subprocess.run(command(kind, slot), cwd=AGENT_ROOT, stdout=out, stderr=out,
                                  timeout=TIMEOUT[kind]).returncode
        except subprocess.TimeoutExpired:
            LOG.warning("[study] %s timed out", kind)
            return 124


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    st = load_state()
    st.setdefault("next", "lesson")
    st.setdefault("practice_slot", 0)
    st.setdefault("lessons", 0)
    st.setdefault("practices", 0)
    LOG.info("[study] loop started: next %s", st["next"])
    while True:
        if not owner_quiet():
            if not st.get("waiting_owner"):
                LOG.info("[study] owner is talking: pause until %d min of quiet", IDLE_MIN)
            st.update(waiting_owner=True, running="", updated=time.time())
            save_state(st)
            time.sleep(OWNER_POLL_S)
            continue
        if is_peak():
            if not st.get("peak"):
                LOG.info("[study] DeepSeek peak hours: no lessons or practice until they end")
            st.update(peak=True, waiting_owner=False, running="", updated=time.time())
            save_state(st)
            time.sleep(PEAK_POLL_S)
            continue
        kind = st["next"]
        st.update(waiting_owner=False, peak=False, running=kind, started=time.time(), updated=time.time())
        save_state(st)
        rc = run_one(kind, st["practice_slot"])
        if kind == "lesson":
            st["lessons"] += 1
            st["next"] = "practice"
        else:
            st["practices"] += 1
            st["practice_slot"] += 1
            st["next"] = "lesson"
        st.update(running="", last=kind, last_rc=rc, finished=time.time(), updated=time.time())
        save_state(st)
        LOG.info("[study] %s finished (rc %s) · lessons %d · practice rounds %d", kind, rc, st["lessons"], st["practices"])
        gap = FAIL_PAUSE_S if rc != 0 else GAP_S
        st.update(resting_until=time.time() + gap)
        save_state(st)
        time.sleep(gap)


if __name__ == "__main__":
    main()
