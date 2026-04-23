import unittest

from scripts.cifar_cdro_budget_utils import (
    cdro_robust_step_compute_be,
    cdro_robust_step_flops,
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


if __name__ == "__main__":
    unittest.main()
