"""Guard against label leakage, refitting preprocessing, and invalid outputs."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import joblib
import numpy as np
import pandas as pd

from src.features import features
from src.model import FreightModel
from src.predict import run
from experiments.benchmark import features as benchmark_features


def fixture():
    return pd.DataFrame({
        "load_id": ["A", "B", "C", "D", "E", "F"],
        "pickup": ["A"] * 6, "delivery": ["B"] * 6,
        "distance": [100., 200., 300., 400., 500., 600.],
        "equipment": ["Dry Van"] * 6,
        "weight": [10000., -1., np.nan, 20000., 30000., 40000.],
        "date": pd.date_range("2025-01-01", periods=6).astype(str),
        "pickup_lat": [35.] * 6, "pickup_lon": [-80.] * 6,
        "delivery_lat": [40.] * 6, "delivery_lon": [-85.] * 6,
        "market_index": [1., np.nan, 1.1, 1.2, 1.0, 1.3],
        "quote_signal": [2.] * 6, "posted_rate": [200., 400., 600., 800., 1000., 1200.],
    })


class PipelineTests(unittest.TestCase):
    def test_selected_model_features_match_recorded_benchmarks(self):
        for reduced in [False, True]:
            expected, expected_cat = benchmark_features(fixture(), reduced=reduced, no_quote=True)
            actual, actual_cat = features(fixture(), reduced=reduced, include_quote=False)
            pd.testing.assert_frame_equal(actual, expected)
            self.assertEqual(actual_cat, expected_cat)

    def test_target_and_id_never_enter_features_and_no_quote_policy(self):
        f = fixture()
        first, _ = features(f, include_quote=False)
        changed = f.assign(posted_rate=999999., load_id="UNRELATED", quote_signal=999.)
        second, _ = features(changed, include_quote=False)
        pd.testing.assert_frame_equal(first, second)
        self.assertFalse({"posted_rate", "load_id", "quote_signal", "quote_total"} & set(first))

    def test_invalid_weight_and_distance(self):
        x, _ = features(fixture(), reduced=True)
        self.assertTrue(x.loc[[1, 2], "weight"].isna().all())
        self.assertEqual(x.weight_missing.sum(), 2)
        f = fixture()
        f.loc[0, "distance"] = 0
        with self.assertRaises(ValueError):
            features(f, reduced=True)

    def test_inference_does_not_refit_training_medians(self):
        f = fixture()
        m = FreightModel(reduced=True, params={"iterations": 3, "depth": 2}).fit(f)
        before = m.medians_.copy()
        unseen = f.iloc[:1].copy().assign(pickup="NEW_CITY", weight=1e10, date="2025-12-01")
        self.assertTrue(np.isfinite(m.predict(unseen)).all())
        pd.testing.assert_series_equal(before, m.medians_)

    def test_prediction_uses_id_mapping_not_row_position(self):
        with TemporaryDirectory() as name:
            p = Path(name)
            f = fixture()
            f.drop(columns="posted_rate").to_csv(p / "validation.csv", index=False)
            template = pd.DataFrame({"load_id": list(reversed(f.load_id)), "predicted_rate": np.nan})
            template.to_csv(p / "validation_predictions_template.csv", index=False)
            december = f.iloc[:2][["pickup", "delivery", "distance", "equipment", "weight", "date"]].copy()
            december["predicted_rate"] = np.nan
            december.to_csv(p / "december_chart_inputs.csv", index=False)
            full = FreightModel(algorithm="ridge", include_quote=False).fit(f)
            reduced = FreightModel(algorithm="ridge", reduced=True).fit(f)
            joblib.dump(full, p / "full_model.joblib")
            joblib.dump(reduced, p / "reduced_model.joblib")
            result, _ = run(p, p, p / "predictions.csv", p / "december.csv")
            self.assertEqual(list(result.load_id), list(reversed(f.load_id)))
            np.testing.assert_allclose(result.predicted_rate, np.round(full.predict(f)[::-1], 2))


if __name__ == "__main__":
    unittest.main()
