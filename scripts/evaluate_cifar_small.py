#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional


def _safe_mean(values: List[float]) -> float:
    return float(mean(values)) if values else float("nan")


def _fmt_float(x: float) -> str:
    if x != x:
        return "nan"
    return f"{x:.6f}"


def _load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _resolve_single_run_dir(run_root: Path) -> Path:
    run_dirs = sorted([p for p in run_root.iterdir() if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No training run directory found under: {run_root}")
    return run_dirs[-1]


def _resolve_latest_eval_dir(run_dir: Path) -> Path:
    eval_root = run_dir / "eval"
    eval_dirs = sorted([p for p in eval_root.iterdir() if p.is_dir()]) if eval_root.exists() else []
    if not eval_dirs:
        raise FileNotFoundError(f"No eval directory found under: {eval_root}")
    return eval_dirs[-1]


def _extract_quick_fid(run_dir: Path) -> Dict[str, Optional[float]]:
    rows = _load_jsonl(run_dir / "quick_eval" / "quick_eval_metrics.jsonl")
    if not rows:
        return {"quick_fid_init": None, "quick_fid_last": None}
    return {
        "quick_fid_init": None if rows[0].get("quick_fid") is None else float(rows[0]["quick_fid"]),
        "quick_fid_last": None if rows[-1].get("quick_fid") is None else float(rows[-1]["quick_fid"]),
    }


def _extract_train_tail(run_dir: Path) -> Dict[str, Optional[float]]:
    rows = _load_jsonl(run_dir / "stats.jsonl")
    if not rows:
        return {"final_kimg": None, "final_loss": None}
    last = rows[-1]
    final_kimg = last.get("Progress/kimg", {}).get("mean")
    final_loss = last.get("Loss/loss", {}).get("mean")
    return {
        "final_kimg": None if final_kimg is None else float(final_kimg),
        "final_loss": None if final_loss is None else float(final_loss),
    }


def _row(seed: int, outdir: Path, prefix: str) -> Dict:
    base_root = outdir / f"{prefix}_baseline_s{seed}"
    robust_root = outdir / f"{prefix}_robust_s{seed}"

    base_run = _resolve_single_run_dir(base_root)
    robust_run = _resolve_single_run_dir(robust_root)
    base_eval = _resolve_latest_eval_dir(base_run)
    robust_eval = _resolve_latest_eval_dir(robust_run)

    base_eval_json = _load_json(base_eval / "evaluation_result.json")
    robust_eval_json = _load_json(robust_eval / "evaluation_result.json")

    base_quick = _extract_quick_fid(base_run)
    robust_quick = _extract_quick_fid(robust_run)
    base_tail = _extract_train_tail(base_run)
    robust_tail = _extract_train_tail(robust_run)

    baseline_fid = float(base_eval_json["fid"])
    robust_fid = float(robust_eval_json["fid"])

    return {
        "seed": int(seed),
        "baseline_run_dir": str(base_run),
        "robust_run_dir": str(robust_run),
        "baseline_eval_dir": str(base_eval),
        "robust_eval_dir": str(robust_eval),
        "baseline_fid": baseline_fid,
        "robust_fid": robust_fid,
        "fid_gap": robust_fid - baseline_fid,
        "robust_beats_baseline": robust_fid < baseline_fid,
        "baseline_quick_fid_init": base_quick["quick_fid_init"],
        "baseline_quick_fid_last": base_quick["quick_fid_last"],
        "robust_quick_fid_init": robust_quick["quick_fid_init"],
        "robust_quick_fid_last": robust_quick["quick_fid_last"],
        "quick_fid_gap_last": (
            None
            if base_quick["quick_fid_last"] is None or robust_quick["quick_fid_last"] is None
            else float(robust_quick["quick_fid_last"] - base_quick["quick_fid_last"])
        ),
        "baseline_final_kimg": base_tail["final_kimg"],
        "robust_final_kimg": robust_tail["final_kimg"],
        "baseline_final_loss": base_tail["final_loss"],
        "robust_final_loss": robust_tail["final_loss"],
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate CIFAR-small baseline vs robust runs across seeds.")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--prefix", type=str, default="cifar_small_once")
    p.add_argument("--seeds", type=str, default="0")
    p.add_argument("--robust-win-ratio-min", type=float, default=0.5)
    p.add_argument("--mean-fid-gap-max", type=float, default=0.0)
    p.add_argument("--min-seed-pass-ratio", type=float, default=0.5)
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--txt-out", type=Path, default=None)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("--seeds must contain at least one seed")

    rows = [_row(seed, args.outdir, args.prefix) for seed in seeds]
    n = len(rows)
    min_seed_pass = max(1, math.ceil(args.min_seed_pass_ratio * n))
    robust_win_count = sum(1 for row in rows if row["robust_beats_baseline"])

    aggregate = {
        "num_seeds": n,
        "min_seed_pass_count": min_seed_pass,
        "robust_win_count": robust_win_count,
        "robust_win_ratio": robust_win_count / n,
        "mean_baseline_fid": _safe_mean([row["baseline_fid"] for row in rows]),
        "mean_robust_fid": _safe_mean([row["robust_fid"] for row in rows]),
        "mean_fid_gap": _safe_mean([row["fid_gap"] for row in rows]),
        "mean_quick_fid_gap_last": _safe_mean(
            [row["quick_fid_gap_last"] for row in rows if row["quick_fid_gap_last"] is not None]
        ),
    }
    aggregate["ready_for_next_stage"] = (
        aggregate["robust_win_ratio"] >= args.robust_win_ratio_min
        and robust_win_count >= min_seed_pass
        and aggregate["mean_fid_gap"] <= args.mean_fid_gap_max
    )

    result = {
        "inputs": {
            "outdir": str(args.outdir),
            "prefix": args.prefix,
            "seeds": seeds,
        },
        "thresholds": {
            "robust_win_ratio_min": args.robust_win_ratio_min,
            "mean_fid_gap_max": args.mean_fid_gap_max,
            "min_seed_pass_ratio": args.min_seed_pass_ratio,
        },
        "per_seed": rows,
        "aggregate": aggregate,
    }

    base_name = f"{args.prefix}_summary"
    json_out = args.json_out or (args.outdir / f"{base_name}.json")
    txt_out = args.txt_out or (args.outdir / f"{base_name}.txt")
    json_out.parent.mkdir(parents=True, exist_ok=True)

    with json_out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    lines = []
    lines.append("CIFAR Small Evaluation")
    lines.append(f"outdir={args.outdir}")
    lines.append(f"prefix={args.prefix} seeds={seeds}")
    lines.append("")
    for row in rows:
        lines.append(
            "seed={seed} fid(base={bf:.4f}, robust={rf:.4f}, gap={gap:.4f}) "
            "robust_win={win} quick_last_gap={qlg} final_kimg(base={bk}, robust={rk})".format(
                seed=row["seed"],
                bf=row["baseline_fid"],
                rf=row["robust_fid"],
                gap=row["fid_gap"],
                win=row["robust_beats_baseline"],
                qlg="nan" if row["quick_fid_gap_last"] is None else f"{row['quick_fid_gap_last']:.4f}",
                bk="nan" if row["baseline_final_kimg"] is None else f"{row['baseline_final_kimg']:.1f}",
                rk="nan" if row["robust_final_kimg"] is None else f"{row['robust_final_kimg']:.1f}",
            )
        )

    lines.append("")
    lines.append(
        "aggregate: ready_for_next_stage={ready} robust_win_ratio={wr:.3f} mean_fid_gap={gap}".format(
            ready=aggregate["ready_for_next_stage"],
            wr=aggregate["robust_win_ratio"],
            gap=_fmt_float(aggregate["mean_fid_gap"]),
        )
    )
    lines.append(
        "means: baseline_fid={bf} robust_fid={rf} quick_last_gap={qg}".format(
            bf=_fmt_float(aggregate["mean_baseline_fid"]),
            rf=_fmt_float(aggregate["mean_robust_fid"]),
            qg=_fmt_float(aggregate["mean_quick_fid_gap_last"]),
        )
    )

    with txt_out.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\n[saved] {json_out}")
    print(f"[saved] {txt_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
