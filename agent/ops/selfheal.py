"""Self-heal watchdog for 星期一 (run as root by r1-selfheal.timer every 2 minutes).

Each check:
  * every watched service that is enabled must be active; one that is not is restarted.
    A crash is "found inactive/failed" or "systemd restarted it on its own" (NRestarts went up).
  * crash loop = CRASH_LIMIT crashes within CRASH_WINDOW. If the code now differs from the newest GOOD
    snapshot (ops.snapshot), roll back to that snapshot (code, config, units - not learned data; the
    current state is saved first as "pre-restore") and restart the r1-* daemons. At most one rollback
    per COOLDOWN, so a snapshot that is itself broken cannot make it loop.
  * all healthy for HEALTHY_FOR and no GOOD snapshot of the code as it is now -> take one, mark it GOOD.
    That is the node the next rollback goes back to.
Events go to run/selfheal_events.jsonl (trimmed); what the owner should hear goes to run/selfheal_notice.txt,
which the orchestrator speaks once at its next start.

    python3 -m ops.selfheal              one check (what the timer runs)
    python3 -m ops.selfheal pause 60     no rollbacks/restarts for 60 minutes (while deploying by hand)
    python3 -m ops.selfheal resume
    python3 -m ops.selfheal status
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from ops import snapshot

LOG = logging.getLogger("selfheal")

SERVICES = ("r1-llm-daemon", "r1-tts-daemon", "r1-orchestrator", "r1-agent-ui")
CRASH_WINDOW = 15 * 60
CRASH_LIMIT = 3
COOLDOWN = 30 * 60
HEALTHY_FOR = 10 * 60
EVENTS_KEEP = 2000


def _run_dir() -> Path:
    return snapshot.AGENT_ROOT / "run"


def _state_path() -> Path:
    return _run_dir() / "selfheal_state.json"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    snapshot._own_like(tmp, path.parent)
    tmp.replace(path)


def load_state() -> dict:
    try:
        st = json.loads(_state_path().read_text(encoding="utf-8"))
        return st if isinstance(st, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(st: dict) -> None:
    _write(_state_path(), json.dumps(st, ensure_ascii=False, indent=1))


def event(kind: str, **kw) -> None:
    rec = {"t": round(time.time(), 1), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": kind, **kw}
    LOG.info("%s %s", kind, json.dumps(kw, ensure_ascii=False))
    path = _run_dir() / "selfheal_events.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-(EVENTS_KEEP - 1):] if path.is_file() else []
    except OSError:
        lines = []
    lines.append(json.dumps(rec, ensure_ascii=False))
    _write(path, "\n".join(lines) + "\n")


def notice(text: str) -> None:
    """Something the owner should hear: the orchestrator says it at its next start, then deletes the file."""
    path = _run_dir() / "selfheal_notice.txt"
    old = ""
    try:
        old = path.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    _write(path, (old + "\n" if old else "") + text + "\n")


# ---------------------------------------------------------------- systemd
def _systemctl(*args: str, timeout: float = 60) -> tuple[int, str]:
    try:
        r = subprocess.run(["systemctl", *args], capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "").strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def service_status(name: str) -> dict:
    _, enabled = _systemctl("is-enabled", name)
    _, active = _systemctl("is-active", name)
    _, n = _systemctl("show", "-p", "NRestarts", "--value", name)
    return {"enabled": enabled == "enabled", "active": active, "restarts": int(n) if n.isdigit() else 0}


def restart(name: str) -> bool:
    _systemctl("reset-failed", name)
    return _systemctl("restart", name, timeout=180)[0] == 0


def r1_daemons() -> list[str]:
    """Long-running r1-* services that are up now (the oneshot ones - chime, timers' jobs - are left alone)."""
    _, out = _systemctl("list-units", "r1-*.service", "--state=active", "--no-legend", "--plain")
    names = [line.split()[0] for line in out.splitlines() if line.strip()]
    keep = []
    for n in names:
        _, typ = _systemctl("show", "-p", "Type", "--value", n)
        if typ != "oneshot" and not n.startswith("r1-selfheal"):
            keep.append(n.removesuffix(".service"))
    return keep


# ---------------------------------------------------------------- the check
class Watchdog:
    """One check. systemd access goes through methods so tests can replace them."""

    status = staticmethod(service_status)
    restart = staticmethod(restart)
    daemons = staticmethod(r1_daemons)

    def daemon_reload(self) -> None:
        _systemctl("daemon-reload")

    def check(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        st = load_state()
        paused = float(st.get("paused_until", 0)) > now
        last = st.setdefault("restarts", {})
        crashes = [t for t in st.get("crashes", []) if now - t < CRASH_WINDOW]
        healthy, report = True, {}
        for name in SERVICES:
            s = self.status(name)
            report[name] = s["active"]
            if not s["enabled"]:
                continue  # switched off on purpose
            prev = last.get(name)
            last[name] = s["restarts"]
            crashed = prev is not None and s["restarts"] > prev
            if crashed:
                event("auto_restarted", service=name, restarts=s["restarts"])
            if s["active"] not in ("active", "activating", "reloading"):
                crashed = True
                if paused:
                    event("down_paused", service=name, state=s["active"])
                else:
                    ok = self.restart(name)
                    event("restart", service=name, state=s["active"], ok=ok)
            if crashed:
                crashes.append(now)
            if s["active"] != "active" or crashed:
                healthy = False
        st["crashes"] = crashes
        action = "ok" if healthy else "unhealthy"

        if len(crashes) >= CRASH_LIMIT and not paused and now - float(st.get("last_rollback", 0)) > COOLDOWN:
            action = self._crash_loop(st, now, len(crashes))
        if healthy:
            st.setdefault("healthy_since", now)
            if now - float(st["healthy_since"]) >= HEALTHY_FOR:
                self._mark_good(st)
        else:
            st.pop("healthy_since", None)
        st["last_check"] = now
        save_state(st)
        LOG.info("check %s crashes(15min)=%d %s%s", action, len(crashes),
                 " ".join(f"{k.removeprefix('r1-')}={v}" for k, v in report.items()), " [paused]" if paused else "")
        return {"action": action, "crashes": len(crashes), "services": report}

    def _crash_loop(self, st: dict, now: float, n: int) -> str:
        current = snapshot.code_hash()
        good = snapshot.latest_good()
        st["last_rollback"] = now  # also when we cannot roll back: don't re-evaluate every 2 minutes
        if good is None or good.code_hash == current:
            why = "没有 GOOD 快照" if good is None else "代码和最近的 GOOD 快照一样，不是改代码引起的"
            event("crash_loop_no_rollback", crashes=n, code_hash=current, why=why)
            notice(f"我刚才 15 分钟里崩溃了 {n} 次，已经重启过服务。{why}，所以没有回退，可能是硬件或模型的问题，需要你看一下日志。")
            return "crash_loop"
        try:
            snap, before = snapshot.restore(good.id)
        except Exception as exc:  # noqa: BLE001
            event("rollback_failed", to=good.id, error=str(exc))
            notice(f"我反复崩溃，想回退到备份节点 {good.id}，但没成功：{exc}。需要你看一下。")
            return "rollback_failed"
        self.daemon_reload()
        names = sorted(set(self.daemons()) | set(SERVICES))
        results = {n_: self.restart(n_) for n_ in names}
        st["restarts"] = {}
        st["crashes"] = []
        event("rollback", to=snap.id, label=snap.label, saved_as=before.id, from_hash=current, to_hash=snap.code_hash,
              restarted=results)
        notice(f"我刚才 15 分钟里崩溃了 {n} 次，已经自动回退到上一个稳定的备份节点 {snap.id}。"
               f"回退前的状态也存好了（{before.id}），想要回来可以再还原。")
        return "rollback"

    def _mark_good(self, st: dict) -> None:
        current = snapshot.code_hash()
        if st.get("good_hash") == current:
            return
        if any(s.good and s.code_hash == current for s in snapshot.list_snapshots()):
            st["good_hash"] = current
            return
        try:
            snap = snapshot.create("auto-good", "连续健康运行 10 分钟")
            snapshot.mark_good(snap.id)
            removed = snapshot.prune()
        except Exception as exc:  # noqa: BLE001
            event("good_snapshot_failed", error=str(exc))
            return
        st["good_hash"] = current
        event("good_snapshot", id=snap.id, code_hash=current, pruned=removed)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [selfheal] %(message)s")
    ap = argparse.ArgumentParser(description="星期一 self-heal watchdog")
    ap.add_argument("cmd", nargs="?", default="check", choices=("check", "pause", "resume", "status"))
    ap.add_argument("minutes", nargs="?", type=float, default=60)
    a = ap.parse_args()
    if a.cmd == "check":
        Watchdog().check()
    elif a.cmd == "pause":
        st = load_state()
        st["paused_until"] = time.time() + a.minutes * 60
        save_state(st)
        event("paused", minutes=a.minutes)
    elif a.cmd == "resume":
        st = load_state()
        st.pop("paused_until", None)
        save_state(st)
        event("resumed")
    else:
        json.dump(load_state(), sys.stdout, ensure_ascii=False, indent=1)
        print()


if __name__ == "__main__":
    os.umask(0o022)
    main()
