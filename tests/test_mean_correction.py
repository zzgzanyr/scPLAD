import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scplad_transport.mean_correction import (
    BUNDLE_FORMAT,
    MeanResponsePredictor,
    apply_mean_correction,
    load_mean_predictor_bundle,
    predict_condition_means,
)


class MeanCorrectionTests(unittest.TestCase):
    def test_population_shift_preserves_centered_cells(self):
        generated = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        target = np.asarray([[2, 4, 6, 8], [1, 3, 5, 7]], dtype=np.float32)
        corrected = apply_mean_correction(generated, target, mixing=0.75)
        expected_mean = 0.25 * generated.mean(axis=1) + 0.75 * target
        np.testing.assert_allclose(corrected.mean(axis=1), expected_mean)
        np.testing.assert_allclose(
            corrected - corrected.mean(axis=1, keepdims=True),
            generated - generated.mean(axis=1, keepdims=True),
        )

    def test_bundle_round_trip_and_prediction(self):
        model = MeanResponsePredictor(
            genes=3, go_features=2, reactome_features=2, block_hidden=2, head_hidden=4
        )
        bundle = {
            "format": BUNDLE_FORMAT,
            "model_config": {
                "genes": 3,
                "go_features": 2,
                "reactome_features": 2,
                "block_hidden": 2,
                "head_hidden": 4,
                "dropout": 0.1,
            },
            "state_dicts": [model.state_dict()],
            "preprocessing": {
                "prior_mean": torch.zeros(4),
                "prior_std": torch.ones(4),
                "control_mean": torch.zeros(3),
                "control_scalar_mean": 0.0,
                "control_scalar_std": 1.0,
                "target_std": 1.0,
            },
            "genes": ["A", "B", "C"],
            "feature_columns": ["go_1", "go_2", "reactome_1", "reactome_2"],
            "correction_mixing": 0.75,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.pt"
            torch.save(bundle, path)
            loaded = load_mean_predictor_bundle(path)
            result = predict_condition_means(loaded, np.zeros((2, 4), dtype=np.float32))
        self.assertEqual(result.shape, (2, 3))
        np.testing.assert_allclose(result[0], result[1])


if __name__ == "__main__":
    unittest.main()
