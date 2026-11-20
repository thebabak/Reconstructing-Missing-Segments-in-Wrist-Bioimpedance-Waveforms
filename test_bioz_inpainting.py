import unittest
import numpy as np
import torch
from bioz_inpainting import (
    DECIMATION, GAPS, PADDING, RATE, WINDOW, UNet, ConditionalVAE,
    GaussianProcess, evaluate, make_examples, preprocess, smoke_data,
)


class LeakageTests(unittest.TestCase):
    def test_hidden_values_cannot_change_input_or_normalization(self):
        raw = smoke_data()['0'][:1].astype(np.float64)
        for seconds in GAPS:
            with self.subTest(seconds=seconds):
                original = make_examples(raw, np.random.default_rng(42), seconds).tensors
                mask = original[2][0, 0].numpy()
                positions = np.flatnonzero(mask)
                start = (PADDING + positions[0]) * DECIMATION
                end = (PADDING + positions[-1] + 1) * DECIMATION
                changed = raw.copy()
                changed[0, start:end] += np.random.default_rng(7).normal(100, 20, end-start)
                altered = make_examples(changed, np.random.default_rng(42), seconds).tensors
                for index in (1, 2, 3):
                    np.testing.assert_array_equal(original[index].numpy(), altered[index].numpy())
                self.assertFalse(np.array_equal(original[0].numpy(), altered[0].numpy()))
                self.assertTrue(np.isfinite(altered[0].numpy()).all())
                self.assertEqual(mask.sum(), round(seconds * RATE))
                self.assertTrue((original[1][0, 0].numpy()[mask] == 0).all())


class ContextTests(unittest.TestCase):
    def test_gap_outputs_reach_one_second_of_both_visible_flanks(self):
        # Positive weights avoid gradient cancellation, exposing structural support.
        torch.set_num_threads(2)
        model = UNet(2).double()
        for parameter in model.parameters():
            torch.nn.init.constant_(parameter, 0.001)
        for seconds in GAPS:
            length = round(seconds * RATE)
            for start in (RATE, (WINDOW - length) // 2, WINDOW - length - RATE):
                for index in (start, start + length // 2, start + length - 1):
                    with self.subTest(seconds=seconds, start=start, index=index):
                        x = torch.zeros(1, 2, WINDOW, dtype=torch.float64, requires_grad=True)
                        derivative = torch.autograd.grad(model(x)[0, 0, index], x)[0][0, 0]
                        self.assertTrue(torch.all(derivative[start-RATE:start] > 0))
                        self.assertTrue(torch.all(derivative[start+length:start+length+RATE] > 0))

    def test_shared_backbone_preserves_length_for_all_model_inputs(self):
        for channels in (2, 3, 4, 18):
            self.assertEqual(UNet(channels)(torch.zeros(2, channels, WINDOW)).shape, (2, 1, WINDOW))


class SamplingTests(unittest.TestCase):
    def test_evaluation_seed_is_independent_of_training_random_stream(self):
        torch.set_num_threads(2)
        torch.manual_seed(1)
        waves = smoke_data()['0'][:2]
        examples = make_examples(waves, np.random.default_rng(42), 2)
        models = (UNet(2).eval(), UNet(4).eval(), UNet(4).eval(),
                  ConditionalVAE().eval(), UNet(3).eval(),
                  GaussianProcess(np.stack([preprocess(raw) for raw in waves])))
        first = list(evaluate(models, examples, torch.device('cpu'), 2, 77))
        torch.manual_seed(999)
        repeated = list(evaluate(models, examples, torch.device('cpu'), 2, 77))
        changed = list(evaluate(models, examples, torch.device('cpu'), 2, 78))
        self.assertEqual(first, repeated)
        self.assertEqual(first[:2], changed[:2])  # Deterministic baselines.
        self.assertNotEqual(first[2:], changed[2:])


if __name__ == '__main__':
    unittest.main()
