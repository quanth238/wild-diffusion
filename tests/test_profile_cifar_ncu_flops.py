import tempfile
import unittest
from pathlib import Path

from scripts.profile_cifar_ncu_flops import (
    DEFAULT_NCU_FLOP_EQUIVALENT_METRICS,
    parse_ncu_raw_csv,
    summarize_ncu_raw_rows,
)


class ProfileCifarNcuFlopsTest(unittest.TestCase):
    def test_parse_and_summarize_ncu_raw_csv(self) -> None:
        metric_ffma = DEFAULT_NCU_FLOP_EQUIVALENT_METRICS[1]
        metric_tensor = DEFAULT_NCU_FLOP_EQUIVALENT_METRICS[3]
        text = (
            "==PROF== status line that should be ignored\n"
            '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
            f'"1","kernel_a","{metric_ffma}","flop","100"\n'
            f'"2","kernel_b","{metric_tensor}","flop","2,048"\n'
            '"3","kernel_c","unselected_metric","unit","999"\n'
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "raw.csv"
            path.write_text(text, encoding="utf-8")
            rows = parse_ncu_raw_csv(path)
        summary = summarize_ncu_raw_rows(rows, flop_equivalent_metrics=DEFAULT_NCU_FLOP_EQUIVALENT_METRICS)
        self.assertEqual(len(rows), 3)
        self.assertEqual(summary["num_selected_metric_rows"], 2)
        self.assertEqual(summary["kernel_instruction_flop_equivalent_per_profiled_range"], 2148.0)


if __name__ == "__main__":
    unittest.main()
