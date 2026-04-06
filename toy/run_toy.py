#!/usr/bin/env python3
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from toy.app.cli import parse_toy_config
    from toy.app.experiment import run_experiment
    from toy.process_title import apply_process_title, build_process_title
else:
    from .app.cli import parse_toy_config
    from .app.experiment import run_experiment
    from .process_title import apply_process_title, build_process_title


_APPLIED_PROCESS_TITLE = apply_process_title()


def parse_args():
    """Backward-compatible parser entrypoint used by old scripts."""

    # Backward-compatible alias used by older scripts.
    return parse_toy_config()


def main() -> None:
    """CLI entrypoint: parse config and execute full toy experiment."""

    cfg = parse_toy_config()
    if _APPLIED_PROCESS_TITLE is None:
        apply_process_title(build_process_title("wdiff", str(cfg.method_version).lower(), cfg.exp_name))
    run_experiment(cfg)


if __name__ == "__main__":
    main()
