#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List


def _load_payload(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if "metrics" not in obj:
        raise ValueError(f"Invalid payload (missing 'metrics'): {path}")
    return obj


def _safe_mean(values: List[float]) -> float:
    return float(mean(values)) if values else float("nan")


def _fmt_float(x: float) -> str:
    if x != x:  # NaN
        return "nan"
    return f"{x:.6f}"


def _kappa_tag(kappa: float) -> str:
    return str(kappa).replace(".", "p")


def _row_from_metrics(seed: int, baseline_payload: Dict, robust_payload: Dict) -> Dict:
    b = baseline_payload["metrics"]
    r = robust_payload["metrics"]

    b_backend = b.get("dataset_debug", {}).get("dataset_backend")
    r_backend = r.get("dataset_debug", {}).get("dataset_backend")
    if b_backend != "toy_gmm" or r_backend != "toy_gmm":
        raise ValueError(
            "evaluate_stageA.py is only valid for the 2D toy GMM protocol; "
            f"got baseline backend={b_backend!r}, robust backend={r_backend!r}"
        )

    b_gate = b["baseline_gate"]
    r_gate = r["baseline_gate"]

    attack_overall = r["constraint_debug"]["attack_gap_windows_heldout"]["overall"]
    attack_high = r["constraint_debug"]["attack_gap_windows_heldout"]["high_noise"]

    curves = r["denoise_debug_heldout_pool"]
    clean_delta = [
        rr - bb
        for rr, bb in zip(curves["robust_on_forward_baseline"], curves["baseline_on_forward_baseline"])
    ]

    sq = r["sample_quality_debug"]
    b_mode = sq["baseline_generated_mode_metrics"]
    r_mode = sq["robust_generated_mode_metrics"]

    frac_near_boundary = r["constraint_debug"]["heldout_rollout_by_step"]["frac_near_boundary"]
    tail_start = (2 * len(frac_near_boundary)) // 3
    tail_sat = _safe_mean(frac_near_boundary[tail_start:])

    return {
        "seed": int(seed),
        "baseline_gate_passed": bool(b_gate["passed"]),
        "robust_gate_passed": bool(r_gate["passed"]),
        "attack_training_executed": bool(r_gate["attack_training_executed"]),
        "attack_win_ratio": float(attack_overall["win_ratio"]),
        "attack_mean_gap": float(attack_overall["mean_attack_gap"]),
        "attack_high_noise_win_ratio": float(attack_high["win_ratio"]),
        "attack_high_noise_mean_gap": float(attack_high["mean_attack_gap"]),
        "clean_mean_gap": _safe_mean(clean_delta),
        "clean_terminal_gap": float(clean_delta[-1]),
        "sample_avg_min_dist_delta": float(r_mode["avg_min_dist_to_mode"] - b_mode["avg_min_dist_to_mode"]),
        "sample_p90_min_dist_delta": float(r_mode["p90_min_dist_to_mode"] - b_mode["p90_min_dist_to_mode"]),
        "tail_constraint_saturation": tail_sat,
    }


def _evaluate(rows: List[Dict], args: argparse.Namespace) -> Dict:
    n = len(rows)
    min_seed_pass = max(1, math.ceil(float(args.min_seed_pass_ratio) * n))

    def attack_pass(row: Dict) -> bool:
        return row["attack_win_ratio"] >= args.attack_win_ratio_min and row["attack_mean_gap"] <= args.attack_mean_gap_max

    def fidelity_pass(row: Dict) -> bool:
        return (
            row["clean_mean_gap"] <= args.clean_mean_gap_max
            and row["clean_terminal_gap"] <= args.clean_terminal_gap_max
            and row["sample_avg_min_dist_delta"] <= args.sample_avg_delta_max
            and row["sample_p90_min_dist_delta"] <= args.sample_p90_delta_max
        )

    def saturation_pass(row: Dict) -> bool:
        return row["tail_constraint_saturation"] <= args.tail_saturation_max

    attack_pass_count = sum(1 for r in rows if attack_pass(r))
    fidelity_pass_count = sum(1 for r in rows if fidelity_pass(r))
    saturation_pass_count = sum(1 for r in rows if saturation_pass(r))

    baseline_gate_all = all(r["baseline_gate_passed"] for r in rows)
    robust_gate_all = all(r["robust_gate_passed"] for r in rows)
    attack_exec_all = all(r["attack_training_executed"] for r in rows)

    attack_majority = attack_pass_count >= min_seed_pass
    fidelity_majority = fidelity_pass_count >= min_seed_pass
    saturation_majority = saturation_pass_count >= min_seed_pass

    ready_for_image = baseline_gate_all and robust_gate_all and attack_exec_all and attack_majority and fidelity_majority and saturation_majority

    agg = {
        "num_seeds": n,
        "min_seed_pass_count": min_seed_pass,
        "attack_pass_count": attack_pass_count,
        "fidelity_pass_count": fidelity_pass_count,
        "saturation_pass_count": saturation_pass_count,
        "baseline_gate_all": baseline_gate_all,
        "robust_gate_all": robust_gate_all,
        "attack_exec_all": attack_exec_all,
        "attack_majority": attack_majority,
        "fidelity_majority": fidelity_majority,
        "saturation_majority": saturation_majority,
        "ready_for_image": ready_for_image,
        "mean_attack_win_ratio": _safe_mean([r["attack_win_ratio"] for r in rows]),
        "mean_attack_gap": _safe_mean([r["attack_mean_gap"] for r in rows]),
        "mean_clean_gap": _safe_mean([r["clean_mean_gap"] for r in rows]),
        "mean_sample_avg_delta": _safe_mean([r["sample_avg_min_dist_delta"] for r in rows]),
        "mean_sample_p90_delta": _safe_mean([r["sample_p90_min_dist_delta"] for r in rows]),
        "mean_tail_saturation": _safe_mean([r["tail_constraint_saturation"] for r in rows]),
        "std_attack_win_ratio": float(pstdev([r["attack_win_ratio"] for r in rows])) if n > 1 else 0.0,
    }
    return agg


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate toy stage-A baseline vs robust runs across seeds.")
    p.add_argument("--outdir", type=Path, required=True, help="Directory containing toy run subfolders.")
    p.add_argument("--prefix", type=str, default="stageA", help="Experiment prefix used in run names.")
    p.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds list, e.g. 0,1,2")
    p.add_argument("--kappa", type=float, default=0.3, help="Kappa used in robust run name.")

    p.add_argument("--attack-win-ratio-min", type=float, default=0.6)
    p.add_argument("--attack-mean-gap-max", type=float, default=0.0)
    p.add_argument("--clean-mean-gap-max", type=float, default=0.03)
    p.add_argument("--clean-terminal-gap-max", type=float, default=0.15)
    p.add_argument("--sample-avg-delta-max", type=float, default=0.03)
    p.add_argument("--sample-p90-delta-max", type=float, default=0.05)
    p.add_argument("--tail-saturation-max", type=float, default=0.995)
    p.add_argument("--min-seed-pass-ratio", type=float, default=0.67)

    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--txt-out", type=Path, default=None)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
    if not seeds:
        raise SystemExit("--seeds must contain at least one seed")

    ktag = _kappa_tag(args.kappa)
    rows = []

    for seed in seeds:
        base_name = f"{args.prefix}_baseline_s{seed}"
        robust_name = f"{args.prefix}_robust_k{ktag}_s{seed}"
        base_metrics = args.outdir / base_name / "metrics.json"
        robust_metrics = args.outdir / robust_name / "metrics.json"

        base_payload = _load_payload(base_metrics)
        robust_payload = _load_payload(robust_metrics)
        rows.append(_row_from_metrics(seed, base_payload, robust_payload))

    agg = _evaluate(rows, args)

    result = {
        "inputs": {
            "outdir": str(args.outdir),
            "prefix": args.prefix,
            "seeds": seeds,
            "kappa": args.kappa,
        },
        "thresholds": {
            "attack_win_ratio_min": args.attack_win_ratio_min,
            "attack_mean_gap_max": args.attack_mean_gap_max,
            "clean_mean_gap_max": args.clean_mean_gap_max,
            "clean_terminal_gap_max": args.clean_terminal_gap_max,
            "sample_avg_delta_max": args.sample_avg_delta_max,
            "sample_p90_delta_max": args.sample_p90_delta_max,
            "tail_saturation_max": args.tail_saturation_max,
            "min_seed_pass_ratio": args.min_seed_pass_ratio,
        },
        "per_seed": rows,
        "aggregate": agg,
    }

    base_name = f"{args.prefix}_summary_k{ktag}"
    json_out = args.json_out or (args.outdir / f"{base_name}.json")
    txt_out = args.txt_out or (args.outdir / f"{base_name}.txt")
    json_out.parent.mkdir(parents=True, exist_ok=True)

    with json_out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    lines = []
    lines.append("Toy Stage-A Evaluation")
    lines.append(f"outdir={args.outdir}")
    lines.append(f"prefix={args.prefix} kappa={args.kappa} seeds={seeds}")
    lines.append("")
    for row in sorted(rows, key=lambda x: x["seed"]):
        lines.append(
            "seed={seed} gate(base/robust)={bg}/{rg} attack_exec={ae} "
            "attack(win={w:.3f},gap={g:.6f}) clean(mean={cm:.6f},term={ct:.6f}) "
            "sample(avg_d={sa:.6f},p90_d={sp:.6f}) sat_tail={sat:.6f}".format(
                seed=row["seed"],
                bg=row["baseline_gate_passed"],
                rg=row["robust_gate_passed"],
                ae=row["attack_training_executed"],
                w=row["attack_win_ratio"],
                g=row["attack_mean_gap"],
                cm=row["clean_mean_gap"],
                ct=row["clean_terminal_gap"],
                sa=row["sample_avg_min_dist_delta"],
                sp=row["sample_p90_min_dist_delta"],
                sat=row["tail_constraint_saturation"],
            )
        )

    lines.append("")
    lines.append(
        "aggregate: ready_for_image={rfi} attack_majority={am} fidelity_majority={fm} saturation_majority={sm}".format(
            rfi=agg["ready_for_image"],
            am=agg["attack_majority"],
            fm=agg["fidelity_majority"],
            sm=agg["saturation_majority"],
        )
    )
    lines.append(
        "means: attack_win={aw} attack_gap={ag} clean_gap={cg} sample_avg_d={sad} sample_p90_d={spd} sat_tail={sat}".format(
            aw=_fmt_float(agg["mean_attack_win_ratio"]),
            ag=_fmt_float(agg["mean_attack_gap"]),
            cg=_fmt_float(agg["mean_clean_gap"]),
            sad=_fmt_float(agg["mean_sample_avg_delta"]),
            spd=_fmt_float(agg["mean_sample_p90_delta"]),
            sat=_fmt_float(agg["mean_tail_saturation"]),
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
