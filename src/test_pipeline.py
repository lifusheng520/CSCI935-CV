"""Run with python -m unittest src.test_pipeline -v (no downloaded weights needed)."""
import unittest
import numpy as np
import torch

from .data import split_records
from .masks import MaskGenerator
from .model import MALG, objective


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_rejected_mask_invariance_and_probability_normalization(self):
        torch.manual_seed(1)
        model = MALG(pretrained=False).eval()
        image = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            a = model(image, torch.zeros(2, 1, 64, 64), torch.zeros(2))
            b = model(image, torch.ones(2, 1, 64, 64), torch.zeros(2))
        self.assertTrue(torch.equal(a["weights"][:, 2], torch.zeros(2)))
        torch.testing.assert_close(a["log_prob"], b["log_prob"])
        torch.testing.assert_close(a["log_prob"].exp().sum(1), torch.ones(2))

    def test_gradients_and_frozen_batchnorm(self):
        torch.manual_seed(2)
        model = MALG(pretrained=False, mask_dropout=0)
        model.set_stage(False)
        model.train()
        bn = model.backbone[0][1]
        before = bn.running_mean.clone()
        image = torch.randn(2, 3, 64, 64)
        out = model(image, torch.ones(2, 1, 64, 64), torch.ones(2))
        objective(out, torch.tensor([0, 1])).backward()
        self.assertIsNotNone(model.threshold.grad)
        self.assertTrue(torch.isfinite(model.threshold.grad))
        self.assertTrue(torch.isfinite(model.scorer.weight.grad).all())
        self.assertTrue(torch.isfinite(model.gate[0].weight.grad).all())
        self.assertIsNotNone(model.heads[2].weight.grad)
        self.assertIsNone(model.backbone[0][0].weight.grad)
        torch.testing.assert_close(before, bn.running_mean)
        model.set_stage(True)
        self.assertTrue(any(p.requires_grad for p in model.backbone[14].parameters()))
        self.assertFalse(any(p.requires_grad for p in model.backbone[6].parameters()))

    def test_mask_dropout_removes_branch(self):
        model = MALG(pretrained=False, mask_dropout=1).train()
        out = model(torch.randn(2, 3, 64, 64), torch.ones(2, 1, 64, 64), torch.ones(2))
        self.assertFalse(out["valid"].any())
        self.assertTrue((out["weights"][:, 2] == 0).all())

    def test_white_mask_keeps_brown_tissue_and_field_fallback(self):
        generator = MaskGenerator(mode="none")
        rgb = np.full((64, 64, 3), 255, np.uint8)
        rgb[10:55, 25:40] = (130, 70, 30)
        mask, quality, _ = generator(rgb, "White")
        self.assertEqual(int(mask[25, 30]), 1)
        self.assertEqual(int(mask[0, 0]), 0)
        self.assertGreater(quality, 0.45)
        rgb[rgb == 255] = 155  # Underexposed paper must still be recognized as background.
        mask, quality, _ = generator(rgb, "White")
        self.assertEqual(int(mask[0, 0]), 0)
        self.assertEqual(int(mask[25, 30]), 1)
        self.assertGreater(quality, 0.45)
        mask, quality, _ = generator(rgb, "Field")
        self.assertEqual(int(mask.sum()), 0)
        self.assertEqual(quality, 0)

    def test_split_is_stratified_disjoint_and_repeatable(self):
        records = [dict(path=f"{d}/{c}/{i}", domain=d, label=c)
                   for d in ("White", "Field") for c in range(5) for i in range(20)]
        a, b = split_records(records, 42), split_records(records, 42)
        self.assertEqual(a, b)
        sets = {k: {r["path"] for r in v} for k,v in a.items()}
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["val"] & sets["test"])
        self.assertFalse(sets["train"] & sets["val"])
        self.assertEqual(len(set.union(*sets.values())), len(records))
        self.assertEqual([len(a[k]) for k in ("train", "val", "test")], [140, 30, 30])
        for rows in a.values():
            self.assertEqual(len({(r["domain"],r["label"]) for r in rows}), 10)


if __name__ == "__main__":
    unittest.main()
