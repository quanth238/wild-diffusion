#!/usr/bin/env python3
"""Collect an optional Nsight Compute kernel FLOP-equivalent diagnostic.

The JSON written by this script is deliberately separate from the analytical
training PFLOP calibration used for paper figures. Nsight Compute metrics are
kernel/instruction diagnostics whose availability and interpretation depend on
GPU architecture, CUDA, driver, and generated kernels.
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = ROOT_DIR / "training-runs" / "compute_diagnostics"
DEFAULT_WORKLOAD_SCRIPT = ROOT_DIR / "scripts" / "cifar_ncu_profile_workload.py"
DEFAULT_NCU_FLOP_EQUIVALENT_METRICS = (
    "derived__smsp__sass_thread_inst_executed_op_hfma_pred_on_x2.sum",
    "derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2.sum",
    "derived__smsp__sass_thread_inst_executed_op_dfma_pred_on_x2.sum",
    "derived__smsp__inst_executed_pipe_tensor_x512.sum",
)
OPERATIONS = ("forward_only", "forward_plus_inputgrad", "forward_plus_parambackward")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Nsight Compute over NVTX-marked CIFAR denoiser primitives and "
            "write a secondary kernel/instruction FLOP-equivalent diagnostic JSON."
        )
    )
    parser.add_argument("--out-json", type=str, default="")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--ncu-bin", type=str, default="")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--workload-script", type=str, default=str(DEFAULT_WORKLOAD_SCRIPT))
    parser.add_argument("--operations", type=str, default=",".join(OPERATIONS))
    parser.add_argument("--metrics", type=str, default=",".join(DEFAULT_NCU_FLOP_EQUIVALENT_METRICS))
    parser.add_argument("--ncu-set", type=str, default="")
    parser.add_argument("--replay-mode", type=str, default="kernel")
    parser.add_argument("--cache-control", type=str, default="none")
    parser.add_argument("--clock-control", type=str, default="none")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--network-pkl", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arch", type=str, default="ddpmpp", choices=["ddpmpp", "ncsnpp", "adm"])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--image-channels", type=int, default=3)
    parser.add_argument("--label-dim", type=int, default=0)
    parser.add_argument("--fp16", type=int, default=1, choices=[0, 1])
    parser.add_argument("--dropout", type=float, default=0.13)
    parser.add_argument("--augment", type=float, default=0.12)
    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--attack-gamma", type=float, default=1.0)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--measure-iters", type=int, default=1)
    return parser.parse_args()


def parse_csv_list(text: str) -> List[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def resolve_ncu_binary(explicit_path: str = "") -> str:
    candidates: List[str] = []
    if str(explicit_path).strip():
        candidates.append(str(explicit_path).strip())
    env_path = os.environ.get("NCU_BIN", "").strip()
    if env_path:
        candidates.append(env_path)
    which_path = shutil.which("ncu")
    if which_path:
        candidates.append(which_path)
    candidates.extend(
        [
            "/usr/local/cuda/bin/ncu",
            "/usr/local/cuda-12.4/bin/ncu",
            "/usr/local/cuda-12.3/bin/ncu",
            "/usr/local/cuda-12.2/bin/ncu",
            "/usr/local/cuda-12.1/bin/ncu",
            "/usr/local/cuda-12.0/bin/ncu",
        ]
    )
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        "Could not find Nsight Compute CLI `ncu`. Pass --ncu-bin or set NCU_BIN."
    )


def capture_text(command: List[str]) -> Dict[str, object]:
    try:
        proc = subprocess.run(command, check=False, text=True, capture_output=True)
    except Exception as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc)}
    return {"returncode": int(proc.returncode), "stdout": proc.stdout, "stderr": proc.stderr}


def query_gpu_summary() -> Dict[str, object]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {"available": False}
    command = [
        nvidia_smi,
        "--query-gpu=name,uuid,driver_version",
        "--format=csv,noheader",
    ]
    result = capture_text(command)
    rows = []
    if result.get("returncode") == 0:
        for line in str(result.get("stdout", "")).splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                rows.append({"name": parts[0], "uuid": parts[1], "driver_version": parts[2]})
    return {"available": bool(rows), "gpus": rows, "command": command}


def parse_numeric_value(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().strip('"')
    if not text or text.lower() in {"n/a", "nan", "inf", "-inf"}:
        return None
    text = text.replace(",", "")
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _first_present(row: Dict[str, str], keys: Iterable[str]) -> str:
    for key in keys:
        if key in row:
            return str(row.get(key, ""))
    return ""


def parse_ncu_raw_csv(path: str | Path) -> List[Dict[str, str]]:
    """Parse the CSV raw page emitted by `ncu --csv --page raw`.

    Nsight Compute may emit status lines before the CSV header. This parser
    starts at the first row containing both `Metric Name` and `Metric Value`.
    """

    raw_path = Path(path)
    if not raw_path.is_file():
        return []
    rows: List[Dict[str, str]] = []
    header: List[str] | None = None
    with raw_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for parsed in csv.reader(handle):
            if not parsed:
                continue
            cells = [cell.strip() for cell in parsed]
            lower = [cell.lower() for cell in cells]
            if header is None:
                if "metric name" in lower and "metric value" in lower:
                    header = cells
                continue
            if len(cells) < len(header):
                cells = cells + [""] * (len(header) - len(cells))
            if len(cells) > len(header):
                cells = cells[: len(header)]
            rows.append(dict(zip(header, cells)))
    return rows


def read_text_tail(path: str | Path, *, max_chars: int = 8000) -> str:
    raw_path = Path(path)
    if not raw_path.is_file():
        return ""
    text = raw_path.read_text(encoding="utf-8", errors="replace")
    return text[-int(max_chars) :]


def ncu_raw_log_flags(text: str) -> Dict[str, bool]:
    return {
        "permission_error": "ERR_NVGPUCTRPERM" in str(text),
        "no_kernels_profiled": "No kernels were profiled" in str(text),
    }


def summarize_ncu_raw_rows(
    rows: List[Dict[str, str]],
    *,
    flop_equivalent_metrics: Iterable[str],
) -> Dict[str, object]:
    selected_metrics = set(str(item) for item in flop_equivalent_metrics)
    metric_summaries: Dict[str, Dict[str, object]] = {}
    for row in rows:
        metric_name = _first_present(row, ("Metric Name", "Metric"))
        if metric_name not in selected_metrics:
            continue
        value = parse_numeric_value(_first_present(row, ("Metric Value", "Value")))
        if value is None:
            continue
        metric = metric_summaries.setdefault(
            metric_name,
            {
                "metric_name": metric_name,
                "metric_unit": _first_present(row, ("Metric Unit", "Unit")),
                "value_sum": 0.0,
                "row_count": 0,
                "kernel_names": [],
            },
        )
        metric["value_sum"] = float(metric["value_sum"]) + float(value)
        metric["row_count"] = int(metric["row_count"]) + 1
        kernel_name = _first_present(row, ("Kernel Name", "Kernel", "Name"))
        if kernel_name and kernel_name not in metric["kernel_names"]:
            metric["kernel_names"].append(kernel_name)

    total = sum(float(item["value_sum"]) for item in metric_summaries.values())
    return {
        "num_raw_metric_rows": int(len(rows)),
        "num_selected_metric_rows": int(sum(int(item["row_count"]) for item in metric_summaries.values())),
        "selected_metric_names": sorted(metric_summaries.keys()),
        "metric_summaries": metric_summaries,
        "kernel_instruction_flop_equivalent_per_profiled_range": float(total) if total > 0.0 else None,
    }


def build_workload_command(args: argparse.Namespace, *, operation: str, nvtx_range: str) -> List[str]:
    command = [
        str(args.python_bin),
        str(Path(args.workload_script).resolve()),
        "--operation",
        str(operation),
        "--nvtx-range",
        str(nvtx_range),
        "--network-pkl",
        str(args.network_pkl),
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--arch",
        str(args.arch),
        "--batch-size",
        str(args.batch_size),
        "--image-size",
        str(args.image_size),
        "--image-channels",
        str(args.image_channels),
        "--label-dim",
        str(args.label_dim),
        "--fp16",
        str(args.fp16),
        "--dropout",
        str(args.dropout),
        "--augment",
        str(args.augment),
        "--sigma-data",
        str(args.sigma_data),
        "--attack-gamma",
        str(args.attack_gamma),
        "--warmup-iters",
        str(args.warmup_iters),
        "--measure-iters",
        str(args.measure_iters),
    ]
    return command


def build_ncu_command(
    args: argparse.Namespace,
    *,
    ncu_bin: str,
    operation: str,
    nvtx_range: str,
    raw_csv_path: Path,
    report_path: Path,
    metrics: List[str],
) -> List[str]:
    nvtx_include = str(nvtx_range)
    if not nvtx_include.endswith("/"):
        nvtx_include = f"{nvtx_include}/"
    command = [
        str(ncu_bin),
        "--target-processes",
        "all",
        "--profile-from-start",
        "on",
        "--nvtx",
        "--nvtx-include",
        nvtx_include,
        "--replay-mode",
        str(args.replay_mode),
        "--cache-control",
        str(args.cache_control),
        "--clock-control",
        str(args.clock_control),
        "--csv",
        "--page",
        "raw",
        "--print-units",
        "base",
        "--print-fp",
        "--log-file",
        str(raw_csv_path),
        "-f",
        "-o",
        str(report_path),
    ]
    if str(args.ncu_set).strip():
        command += ["--set", str(args.ncu_set).strip()]
    else:
        command += ["--metrics", ",".join(metrics)]
    command += build_workload_command(args, operation=operation, nvtx_range=nvtx_range)
    return command


def build_tag(args: argparse.Namespace) -> str:
    if str(args.tag).strip():
        return str(args.tag).strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fp_label = "fp16" if int(args.fp16) else "fp32"
    return f"cifar10_32x32_{args.arch}_wdroedm_{fp_label}_b{int(args.batch_size)}_ncu_{stamp}"


def main() -> None:
    args = parse_args()
    operations = parse_csv_list(args.operations)
    unknown = [operation for operation in operations if operation not in OPERATIONS]
    if unknown:
        raise ValueError(f"Unsupported operation(s): {unknown}. Supported: {OPERATIONS}")
    if not operations:
        raise ValueError("At least one operation is required.")

    metrics = parse_csv_list(args.metrics)
    if not metrics and not str(args.ncu_set).strip():
        raise ValueError("Pass --metrics or --ncu-set.")

    ncu_bin = resolve_ncu_binary(args.ncu_bin)
    ncu_version = capture_text([ncu_bin, "--version"])
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    tag = build_tag(args)
    out_json = Path(args.out_json).resolve() if str(args.out_json).strip() else outdir / f"{tag}.json"

    payload: Dict[str, object] = {
        "format": "image_ncu_hardware_flop_diagnostic_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "semantics": {
            "role": "secondary_hardware_kernel_diagnostic",
            "not_primary_method_compute": True,
            "primary_method_compute_remains": "analytical_training_pflops_from_forward_profiler_calibration",
            "definition": (
                "Sum of selected Nsight Compute derived instruction metrics over kernels inside the "
                "NVTX-marked range. This is a hardware/kernel diagnostic, not a literature-normalized "
                "training FLOP convention and not exact CUDA source-operation FLOPs."
            ),
            "selected_metric_interpretation": (
                "The default metrics are Nsight Compute derived instruction-equivalent counts for "
                "half/single/double FMA operations and tensor-pipe instructions using Nsight's x512 "
                "derived conversion where available. The `.sum` suffix is used so kernel rows can be "
                "added over the profiled NVTX range."
            ),
        },
        "ncu": {
            "binary": ncu_bin,
            "version_stdout": str(ncu_version.get("stdout", "")).strip(),
            "version_stderr": str(ncu_version.get("stderr", "")).strip(),
            "version_returncode": ncu_version.get("returncode"),
            "metrics": metrics,
            "ncu_set": str(args.ncu_set).strip() or None,
            "replay_mode": str(args.replay_mode),
            "cache_control": str(args.cache_control),
            "clock_control": str(args.clock_control),
        },
        "hardware": query_gpu_summary(),
        "workload": {
            "model_kind": "image_edmprecond",
            "training_objective": "edm",
            "image_backbone": str(args.arch),
            "arch": str(args.arch),
            "precond": "wdroedm",
            "batch_size": int(args.batch_size),
            "image_channels": int(args.image_channels),
            "image_size": int(args.image_size),
            "label_dim": int(args.label_dim),
            "use_fp16": bool(args.fp16),
            "dropout": float(args.dropout),
            "augment_p": float(args.augment),
            "sigma_data": float(args.sigma_data),
            "attack_gamma": float(args.attack_gamma),
            "network_pkl": (str(Path(args.network_pkl).resolve()) if str(args.network_pkl).strip() else None),
            "warmup_iters": int(args.warmup_iters),
            "measure_iters": int(args.measure_iters),
        },
        "operations": {},
    }

    had_error = False
    for operation in operations:
        nvtx_range = f"{tag}_{operation}"
        raw_csv_path = outdir / f"{tag}_{operation}_raw.csv"
        report_path = outdir / f"{tag}_{operation}.ncu-rep"
        command = build_ncu_command(
            args,
            ncu_bin=ncu_bin,
            operation=operation,
            nvtx_range=nvtx_range,
            raw_csv_path=raw_csv_path,
            report_path=report_path,
            metrics=metrics,
        )
        op_payload: Dict[str, object] = {
            "operation": operation,
            "nvtx_range": nvtx_range,
            "raw_csv_path": str(raw_csv_path),
            "ncu_report_path": str(report_path),
            "command": command,
        }
        if args.dry_run:
            op_payload["status"] = "dry_run"
            op_payload["returncode"] = None
        else:
            print(f"[ncu-diagnostic] profiling {operation} -> {raw_csv_path}", flush=True)
            proc = subprocess.run(command, check=False, text=True, capture_output=True)
            op_payload["status"] = "ok" if proc.returncode == 0 else "failed"
            op_payload["returncode"] = int(proc.returncode)
            op_payload["stdout_tail"] = proc.stdout[-4000:]
            op_payload["stderr_tail"] = proc.stderr[-4000:]
            if proc.returncode != 0:
                had_error = True
            raw_log_tail = read_text_tail(raw_csv_path)
            raw_log_flags = ncu_raw_log_flags(raw_log_tail)
            rows = parse_ncu_raw_csv(raw_csv_path)
            summary = summarize_ncu_raw_rows(rows, flop_equivalent_metrics=metrics)
            op_payload["summary"] = summary
            op_payload["raw_log_tail"] = raw_log_tail
            op_payload["ncu_permission_error"] = bool(raw_log_flags["permission_error"])
            op_payload["ncu_no_kernels_profiled"] = bool(raw_log_flags["no_kernels_profiled"])
            value = summary.get("kernel_instruction_flop_equivalent_per_profiled_range")
            op_payload["kernel_instruction_flop_equivalent_per_batch"] = (
                None if value is None else float(value) / float(max(int(args.measure_iters), 1))
            )
            if raw_log_flags["permission_error"]:
                op_payload["status"] = "permission_denied"
                had_error = True
            elif raw_log_flags["no_kernels_profiled"]:
                op_payload["status"] = "no_kernels_profiled"
                had_error = True
            elif int(summary.get("num_selected_metric_rows", 0)) <= 0:
                op_payload["status"] = "no_selected_metrics"
                had_error = True
        payload["operations"][operation] = op_payload
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if had_error and not args.continue_on_error:
            break

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[ncu-diagnostic] wrote {out_json}", flush=True)
    if had_error and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
