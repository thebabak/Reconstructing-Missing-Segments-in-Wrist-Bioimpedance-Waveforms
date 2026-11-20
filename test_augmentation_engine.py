import unittest
import numpy as np
import torch
from bioz_inpainting import smoke_data, make_examples as original_examples, train as original_train
from augmentation_engine import make_examples, train
from validated_benchmark import CNN_NAMES,predict


class ExecutionEquivalenceTests(unittest.TestCase):
    def test_batched_preprocessing_is_bitwise_equal_and_preserves_mask_rng(self):
        waves = smoke_data()['0']
        for gap in (None,.5,1,2):
            first, second = np.random.default_rng(2026), np.random.default_rng(2026)
            a, b = original_examples(waves,first,gap), make_examples(waves,second,gap)
            for x, y in zip(a.tensors,b.tensors):
                np.testing.assert_array_equal(x.numpy(),y.numpy())
            self.assertEqual(first.bit_generator.state, second.bit_generator.state)

    def check_training(self, device):
        torch.set_num_threads(2)
        waves = smoke_data()['0'][:4]
        original_states = {}
        def capture(epoch, models):
            original_states[epoch] = [{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for m in models]
            return False
        torch.manual_seed(2026)
        original = original_train(waves,3,device,np.random.default_rng(7),model_callback=capture)
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device) if device.type=='cuda' else None
        torch.manual_seed(2026)
        optimized = train(waves,3,device,np.random.default_rng(7))
        self.assertTrue(torch.equal(torch.random.get_rng_state(),cpu_rng))
        if cuda_rng is not None: self.assertTrue(torch.equal(torch.cuda.get_rng_state(device),cuda_rng))
        for index, model in enumerate(optimized):
            for key, value in model.state_dict().items():
                # Repeated original CUDA GAN fits already differ by up to 6e-5.
                tolerance = 1e-4 if device.type=='cuda' else 1e-7
                torch.testing.assert_close(value.cpu(),original_states[3][index][key],atol=tolerance,rtol=1e-5)
        stops = dict(zip(CNN_NAMES,(1,3,3,1,1)))
        torch.manual_seed(2026)
        optimized = train(waves,3,device,np.random.default_rng(7),active=lambda m,e:e<=stops[m])
        self.assertTrue(torch.equal(torch.random.get_rng_state(),cpu_rng))
        if cuda_rng is not None: self.assertTrue(torch.equal(torch.cuda.get_rng_state(device),cuda_rng))
        for index, (name,model) in enumerate(zip(CNN_NAMES,optimized)):
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value.cpu(),original_states[stops[name]][index][key],atol=tolerance,rtol=1e-5)
        for index, (name,model) in enumerate(zip(CNN_NAMES,original[:5])):
            model.load_state_dict(original_states[stops[name]][index])
        examples=original_examples(waves,np.random.default_rng(42),2)
        a=predict(dict(zip(CNN_NAMES,original[:5])),examples,device,12,77)
        b=predict(dict(zip(CNN_NAMES,optimized)),examples,device,12,77)
        for name in CNN_NAMES:np.testing.assert_allclose(a[name],b[name],atol=2e-4,rtol=1e-4)

    def test_optimized_training_matches_full_loop_cpu(self):
        self.check_training(torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_optimized_training_matches_full_loop_cuda(self):
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        self.check_training(torch.device('cuda'))


if __name__=='__main__':
    unittest.main()
