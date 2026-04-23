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
