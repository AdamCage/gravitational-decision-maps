import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from gdm.evidence import PROJECT_ROOT, _compare_metrics, _episteme, freeze, run_or_export


class MetricComparisonTests(unittest.TestCase):
    def test_secondary_disagreements_and_missing_metrics_are_detected(self):
        comparison = _compare_metrics({"primary_gain": 0.012, "coverage": 0.9, "row_count": 20},
                                      {"primary_gain": 0.012 + 1e-15, "coverage": 0.8, "other": 20})
        self.assertTrue(comparison["primary_gain"]["agrees"])
        for name in ("coverage", "row_count", "other"):
            self.assertFalse(comparison[name]["agrees"])


class FreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkout = PROJECT_ROOT.parent / "EpistemeOS"
        if not (checkout / ".git").exists():
            raise unittest.SkipTest("Pinned EpistemeOS sibling checkout is absent")
        _episteme(checkout)

    def test_pre_registration_does_not_access_outcomes_and_refuses_source_drift(self):
        with TemporaryDirectory(prefix="gdm-test-freeze-") as temporary:
            root = Path(temporary)
            project = root / "study"
            project.mkdir()
            for name in ("pyproject.toml", "protocol.json", "uv.lock"):
                (project / name).write_bytes((PROJECT_ROOT / name).read_bytes())
            for name in ("study_worker.py", "study_reanalysis.py"):
                (project / name).write_text('raise RuntimeError("Run unexpectedly")\n', encoding="utf-8")
            state = root / "state"
            record = freeze(state, workers=1, project=project)
            self.assertEqual(json.loads((state / "frozen.json").read_text(encoding="utf-8")), record)
            self.assertEqual(freeze(state, workers=1, project=project), record)
            with self.assertRaisesRegex(ValueError, "no primary execution job"):
                run_or_export(state, root / "out", "synthetic", execute=False, project=project)
            with self.assertRaisesRegex(ValueError, "Worker count already locked"):
                freeze(state, workers=2, project=project)
            (project / "study_worker.py").write_text("print('drift')\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Changes since study freeze: study_worker.py"):
                run_or_export(state, root / "out", "synthetic", execute=False, project=project)


if __name__ == "__main__":
    unittest.main()
