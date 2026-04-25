import json
import tempfile
import unittest
from pathlib import Path

from scripts.cifar_cdro_budget_utils import (
    cdro_robust_step_compute_be,
    cdro_robust_step_flops,
    hardware_flop_diagnostic_metadata_fields,
)


class CifarCdroBudgetUtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.flop_calibration = {
            "available": True,
            "forward_flops": 10.0,
            "inputgrad_flops": 20.0,
            "parambackward_flops": 30.0,
        }

    def test_rho_zero_skips_attack_construction_cost(self) -> None:
        self.assertEqual(
            cdro_robust_step_compute_be(
                n_steps_path=32,
                attack_num_steps=1,
                outer_attack_weight=1.0,
                outer_clean_weight=0.0,
                total_budget_rho=0.0,
            ),
            32.0,
        )

    def test_positive_rho_keeps_attack_construction_cost(self) -> None:
        self.assertEqual(
            cdro_robust_step_compute_be(
                n_steps_path=32,
                attack_num_steps=1,
                outer_attack_weight=1.0,
                outer_clean_weight=0.0,
                total_budget_rho=1e-5,
            ),
            64.0,
        )

    def test_rho_zero_skips_attack_construction_flops(self) -> None:
        self.assertEqual(
            cdro_robust_step_flops(
                calibration=self.flop_calibration,
                n_steps_path=4,
                attack_num_steps=1,
                outer_attack_weight=1.0,
                outer_clean_weight=0.0,
                total_budget_rho=0.0,
            ),
            120.0,
        )

    def test_positive_rho_keeps_attack_construction_flops(self) -> None:
        self.assertEqual(
            cdro_robust_step_flops(
                calibration=self.flop_calibration,
                n_steps_path=4,
                attack_num_steps=1,
                outer_attack_weight=1.0,
                outer_clean_weight=0.0,
                total_budget_rho=1e-5,
            ),
            200.0,
        )

    def test_hardware_flop_diagnostic_metadata_fields(self) -> None:
        payload = {
            "format": "image_ncu_hardware_flop_diagnostic_v1",
            "semantics": {
                "role": "secondary_hardware_kernel_diagnostic",
                "not_primary_method_compute": True,
                "definition": "test definition",
            },
            "ncu": {"version_stdout": "Nsight Compute version test"},
            "operations": {
                "forward_only": {"kernel_instruction_flop_equivalent_per_batch": 10.0},
                "forward_plus_inputgrad": {"kernel_instruction_flop_equivalent_per_batch": 20.0},
                "forward_plus_parambackward": {"kernel_instruction_flop_equivalent_per_batch": 30.0},
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ncu.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            fields = hardware_flop_diagnostic_metadata_fields(str(path))
        self.assertEqual(fields["train_hardware_flop_diagnostic_role"], "secondary_hardware_kernel_diagnostic")
        self.assertTrue(fields["train_hardware_flop_diagnostic_not_primary"])
        self.assertEqual(fields["train_hardware_forward_flop_equivalent_per_batch"], 10.0)
        self.assertEqual(fields["train_hardware_inputgrad_flop_equivalent_per_batch"], 20.0)
        self.assertEqual(fields["train_hardware_parambackward_flop_equivalent_per_batch"], 30.0)

    def test_hardware_flop_diagnostic_requires_not_primary_marker(self) -> None:
        payload = {
            "format": "image_ncu_hardware_flop_diagnostic_v1",
            "semantics": {"role": "secondary_hardware_kernel_diagnostic"},
            "operations": {},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ncu.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not_primary_method_compute=true"):
                hardware_flop_diagnostic_metadata_fields(str(path))


if __name__ == "__main__":
    unittest.main()
