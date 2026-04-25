import json
import tempfile
import unittest
from pathlib import Path

from toy.compute_accounting import (
    denoiser_flops,
    denoiser_flops_from_count_record,
    flops_to_pflops,
    flops_to_tflops,
    load_flop_calibration,
    load_weighted_compute_calibration,
)


class ComputeAccountingFlopsTest(unittest.TestCase):
    def test_load_flop_calibration_from_explicit_values(self) -> None:
        calibration = load_flop_calibration(
            forward_flops=10.0,
            inputgrad_flops=20.0,
            parambackward_flops=30.0,
        )
        self.assertTrue(calibration["available"])
        self.assertEqual(calibration["source"], "explicit_cli")
        self.assertEqual(calibration["flop_cost_source"], "explicit_cli")
        self.assertEqual(
            denoiser_flops(
                n_fwd=1.0,
                n_fwd_inputgrad=2.0,
                n_fwd_parambackward=3.0,
                calibration=calibration,
            ),
            140.0,
        )

    def test_load_flop_calibration_from_json(self) -> None:
        payload = {
            "flops": {
                "forward_only": {"per_batch_flops": 11.0},
                "forward_plus_inputgrad": {"per_batch_flops": 22.0},
                "forward_plus_parambackward": {"per_batch_flops": 33.0},
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flops.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            calibration = load_flop_calibration(calibration_path=str(path))
        self.assertTrue(calibration["available"])
        self.assertEqual(calibration["forward_flops"], 11.0)
        self.assertEqual(calibration["inputgrad_flops"], 22.0)
        self.assertEqual(calibration["parambackward_flops"], 33.0)
        self.assertEqual(calibration["flop_cost_source"], "profiler_supported_operator_flops")

    def test_image_flop_calibration_uses_analytical_training_costs(self) -> None:
        payload = {
            "format": "image_flop_calibration_v1",
            "flops": {
                "forward_only": {"per_batch_flops": 11.0},
                "forward_plus_inputgrad": {"per_batch_flops": 12.0},
                "forward_plus_parambackward": {"per_batch_flops": 13.0},
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flops.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            calibration = load_flop_calibration(calibration_path=str(path))
        self.assertTrue(calibration["available"])
        self.assertEqual(calibration["forward_flops"], 11.0)
        self.assertEqual(calibration["inputgrad_flops"], 22.0)
        self.assertEqual(calibration["parambackward_flops"], 33.0)
        self.assertEqual(
            calibration["flop_cost_source"],
            "analytical_training_from_profiler_forward_legacy_payload",
        )

    def test_training_flops_group_overrides_profiler_flops(self) -> None:
        payload = {
            "format": "image_flop_calibration_v1",
            "flops": {
                "forward_only": {"per_batch_flops": 11.0},
                "forward_plus_inputgrad": {"per_batch_flops": 12.0},
                "forward_plus_parambackward": {"per_batch_flops": 13.0},
            },
            "training_flops": {
                "source": "analytical_training_from_profiler_forward",
                "definition": "test convention",
                "forward_only": {"per_batch_flops": 101.0},
                "forward_plus_inputgrad": {"per_batch_flops": 202.0},
                "forward_plus_parambackward": {"per_batch_flops": 303.0},
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flops.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            calibration = load_flop_calibration(calibration_path=str(path))
        self.assertEqual(calibration["forward_flops"], 101.0)
        self.assertEqual(calibration["inputgrad_flops"], 202.0)
        self.assertEqual(calibration["parambackward_flops"], 303.0)
        self.assertEqual(calibration["flop_cost_group"], "training_flops")

    def test_weighted_calibration_rejects_flop_payload_ratios(self) -> None:
        payload = {
            "format": "image_flop_calibration_v1",
            "ratios": {
                "inputgrad_alpha": 1.01,
                "parambackward_beta": 1.02,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flops.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "cannot be used as weighted-compute"):
                load_weighted_compute_calibration(calibration_path=str(path))

    def test_denoiser_flops_from_count_record_honors_override_forward_count(self) -> None:
        calibration = load_flop_calibration(
            forward_flops=10.0,
            inputgrad_flops=20.0,
            parambackward_flops=30.0,
        )
        count_record = {
            "n_fwd": 5.0,
            "n_fwd_inputgrad": 2.0,
            "n_fwd_parambackward": 1.0,
        }
        self.assertEqual(
            denoiser_flops_from_count_record(
                count_record=count_record,
                calibration=calibration,
                override_n_fwd=1.0,
            ),
            80.0,
        )

    def test_flop_unit_conversions(self) -> None:
        self.assertEqual(flops_to_tflops(3.0e12), 3.0)
        self.assertEqual(flops_to_pflops(4.0e15), 4.0)


if __name__ == "__main__":
    unittest.main()
