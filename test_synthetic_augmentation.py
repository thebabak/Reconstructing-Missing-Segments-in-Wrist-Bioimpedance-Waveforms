import unittest
import numpy as np
from bioz_inpainting import smoke_data, make_examples
from synthetic_augmentation import SyntheticBioZ, training_set, check_split, NATIVE_LENGTH


class SyntheticAugmentationTests(unittest.TestCase):
    def test_generator_is_repeatable_produces_new_native_waves_and_supports_gap_preprocessing(self):
        real = smoke_data()['0'][:3]
        generator = SyntheticBioZ().fit(real)
        first, donors = generator.sample(3, 42)
        repeated, repeated_donors = generator.sample(3, 42)
        changed, _ = generator.sample(3, 43)
        np.testing.assert_array_equal(first, repeated)
        np.testing.assert_array_equal(donors, repeated_donors)
        self.assertEqual(first.shape, (3, NATIVE_LENGTH))
        self.assertFalse(np.array_equal(first, changed))
        self.assertFalse(np.array_equal(first, real[donors]))
        examples = make_examples(first, np.random.default_rng(8), 2)
        self.assertTrue(all(np.isfinite(v.numpy()).all() for v in examples.tensors))
        self.assertTrue(all(v > 0 for v in examples.tensors[-1]))

    def test_comparison_matches_epoch_size_without_relabeling_real_as_synthetic(self):
        real = np.ones((2, NATIVE_LENGTH)); synthetic = np.zeros_like(real)
        control = training_set(real, synthetic, 'real_only_matched')
        mixed = training_set(real, synthetic, 'real_plus_synthetic')
        self.assertEqual(control.shape, mixed.shape)
        np.testing.assert_array_equal(control[:2], mixed[:2])
        np.testing.assert_array_equal(control[2:], real)
        np.testing.assert_array_equal(mixed[2:], synthetic)

    def test_split_rejects_test_or_validation_in_inner_training(self):
        check_split(['3','4','5','6'], '2', ['2','3','4','5','6'], '1')
        with self.assertRaises(AssertionError):
            check_split(['1','4','5','6'], '2', ['1','2','4','5','6'], '1')
        with self.assertRaises(AssertionError):
            check_split(['2','4','5','6'], '2', ['2','4','5','6'], '1')


if __name__ == '__main__':
    unittest.main()
