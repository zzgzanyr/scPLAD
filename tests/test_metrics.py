import unittest

import numpy as np

from scplad_transport.metrics import (
    correlation_structure,
    expression_range_coverage,
    response_auprc,
    topk_overlap,
)


class MetricDefinitionTests(unittest.TestCase):
    def test_erc_is_coverage_of_true_width_not_interval_iou(self):
        true = np.array([[0.0], [1.0], [2.0]])
        pred = np.array([[-1.0], [1.0], [3.0]])
        score, per_gene = expression_range_coverage(true, pred, 0.0, 1.0)
        self.assertAlmostEqual(score, 1.0)
        self.assertAlmostEqual(float(per_gene[0]), 1.0)

    def test_topk_overlap_and_auprc_are_perfect_for_matching_ranking(self):
        true_delta = np.array([4.0, 3.0, 0.2, 0.1])
        pred_delta = np.array([-8.0, 7.0, 0.3, 0.0])
        self.assertEqual(topk_overlap(true_delta, pred_delta, 2), 1.0)
        self.assertEqual(response_auprc(true_delta, pred_delta, 2), 1.0)

    def test_csa_identity(self):
        rng = np.random.default_rng(7)
        cells = rng.normal(size=(80, 8))
        pcc, spearman = correlation_structure(cells, cells.copy(), np.arange(8))
        self.assertAlmostEqual(pcc, 1.0)
        self.assertAlmostEqual(spearman, 1.0)


if __name__ == "__main__":
    unittest.main()
