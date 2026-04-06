import os
from typing import Dict, Optional


PROCESS_TITLE_ENV = "WILD_DIFFUSION_PROC_TITLE"
_PROCESS_TITLE_MAX_LEN = 120


def build_process_title(*parts: object, max_len: int = _PROCESS_TITLE_MAX_LEN) -> Optional[str]:
    tokens = []
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        normalized = text.replace(os.sep, "_").replace(" ", "_")
        tokens.append(normalized)
    if not tokens:
        return None
    title = ":".join(tokens)
    if len(title) <= max_len:
        return title
    head = ":".join(tokens[:-1])
    if not head:
        return title[-max_len:]
    tail_budget = max(1, max_len - len(head) - 1)
    return f"{head}:{tokens[-1][-tail_budget:]}"


def requested_process_title(env: Optional[Dict[str, str]] = None) -> Optional[str]:
    source = os.environ if env is None else env
    title = str(source.get(PROCESS_TITLE_ENV, "")).strip()
    if not title:
        return None
    return build_process_title(title)


def child_process_env(*, proc_title: Optional[str]) -> Dict[str, str]:
    env = os.environ.copy()
    normalized = build_process_title(proc_title) if proc_title else None
    if normalized:
        env[PROCESS_TITLE_ENV] = normalized
    else:
        env.pop(PROCESS_TITLE_ENV, None)
    return env


def apply_process_title(title: Optional[str] = None) -> Optional[str]:
    resolved = build_process_title(title) if title else requested_process_title()
    if not resolved:
        return None
    try:
        from setproctitle import setproctitle
    except Exception:
        return None
    try:
        setproctitle(resolved)
    except Exception:
        return None
    return resolved
