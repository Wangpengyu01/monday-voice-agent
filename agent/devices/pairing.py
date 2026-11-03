"""One-time pairing codes for the device page: 星期一 says the code out loud, the user types it in."""
from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path

CODE_TTL_S = 10 * 60


def new_code(path: str | Path) -> str:
    code = f"{secrets.randbelow(10000):04d}"
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"code": code, "expires": time.time() + CODE_TTL_S}), encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(p)
    return code


def check_code(path: str | Path, code: str) -> bool:
    """True once per code: a used code is burned."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if time.time() > float(data.get("expires", 0)):
        return False
    if not secrets.compare_digest(str(data.get("code", "")), (code or "").strip()):
        return False
    p.unlink(missing_ok=True)
    return True
