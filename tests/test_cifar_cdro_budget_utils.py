import unittest

from scripts.cifar_cdro_budget_utils import cdro_robust_step_compute_be


class CifarCdroBudgetUtilsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
