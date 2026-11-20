import unittest
import numpy as np
import torch
from bioz_inpainting import GaussianProcess,make_examples,preprocess,smoke_data,peak_stats,PADDING,DECIMATION
from validated_benchmark import subject_split,EpochSelector,classical_predictions,new_saits,predict

class ValidatedProtocolTests(unittest.TestCase):
 def test_no_outer_subject_enters_selection(self):
  subjects=[str(i) for i in range(1,7)]
  for outer in subjects:
   validations=[]
   for seed in range(2026,2031):
    inner,val,full=subject_split(subjects,outer,seed)
    self.assertEqual(len(inner),4);self.assertNotIn(outer,inner);self.assertNotIn(val,inner)
    self.assertNotEqual(val,outer);self.assertEqual(set(inner+[val]),set(full));validations.append(val)
   self.assertEqual(set(validations),set(subjects)-{outer})
 def test_selector_locks_after_patience(self):
  selector=EpochSelector(('m',),patience=2)
  selector.update(5,{'m':3});selector.update(10,{'m':2});selector.update(15,{'m':2.1})
  self.assertTrue(selector.update(20,{'m':2.2}));selector.update(25,{'m':1})
  self.assertEqual(selector.epochs['m'],10)
 def test_baselines_preserve_context_and_gp_mean_is_zero_noise_draw(self):
  waves=smoke_data()['0'][:2];examples=make_examples(waves,np.random.default_rng(42),2)
  gp=GaussianProcess(np.stack([preprocess(r) for r in waves]))
  outputs=classical_predictions(waves,examples,gp)
  context=examples.tensors[1][:,0].numpy();gap=examples.tensors[2][:,0].numpy()
  for output in outputs.values():
   self.assertTrue(np.isfinite(output).all());np.testing.assert_allclose(output[~gap],context[~gap],atol=1e-6)
  class ZeroNoise:
   def standard_normal(self,n):return np.zeros(n)
  np.testing.assert_allclose(gp.mean(context[0],gap[0]),gp.sample(context[0],gap[0],ZeroNoise()))
 def test_peak_tolerance_and_prominence_are_used(self):
  ref=np.zeros(1000);out=ref.copy();gap=np.zeros(1000,dtype=bool);gap[300:700]=True
  ref[500]=1;out[512]=1
  self.assertEqual(peak_stats(ref,out,gap,.05,.35)[1:],(0,1,1))
  self.assertEqual(peak_stats(ref,out,gap,.15,.35)[1:],(1,0,0))
  out[512]=.25
  self.assertEqual(peak_stats(ref,out,gap,.15,.35)[1:],(0,1,0))
  self.assertEqual(peak_stats(ref,out,gap,.15,.2)[1:],(1,0,0))
 def test_new_baselines_cannot_use_hidden_test_values(self):
  test=smoke_data()['0'][:1]
  training=smoke_data()['1']
  gp=GaussianProcess(np.stack([preprocess(r) for r in training]))
  for seconds in (.5,1,2):
   original=make_examples(test,np.random.default_rng(42),seconds)
   positions=np.flatnonzero(original.tensors[2][0,0].numpy())
   start=(PADDING+positions[0])*DECIMATION;end=(PADDING+positions[-1]+1)*DECIMATION
   perturbed=test.copy();perturbed[0,start:end]+=100
   altered=make_examples(perturbed,np.random.default_rng(42),seconds)
   a=classical_predictions(test,original,gp);b=classical_predictions(perturbed,altered,gp)
   for name in a:np.testing.assert_array_equal(a[name],b[name])
 def test_saits_inference_preserves_observations(self):
  torch.set_num_threads(2)
  examples=make_examples(smoke_data()['0'][:2],np.random.default_rng(42),2)
  model=new_saits(torch.device('cpu')).eval()
  output=predict({'saits':model},examples,torch.device('cpu'),1,42)['saits']
  mask=examples.tensors[2][:,0].numpy();context=examples.tensors[1][:,0].numpy()
  np.testing.assert_array_equal(output[~mask],context[~mask]);self.assertTrue(np.isfinite(output).all())

if __name__=='__main__':unittest.main()
