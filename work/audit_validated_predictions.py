"""Independent replay of a completed checkpoint against its stored predictions."""
import csv,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bioz_inpainting import UNet,ConditionalVAE,load_trials,make_examples
from validated_benchmark import new_saits,predict,METHODS,peak_stats

ROOT=Path(__file__).resolve().parents[1]
def audit(subject="1",seed=2026):
 fold=int(subject)-1
 torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 manifest=json.loads((ROOT/'outputs/bioz_metrics_validated.run.json').read_text())
 assert [seed,subject] in manifest['completed_folds']
 checkpoint=torch.load(ROOT/f'outputs/bioz_metrics_validated_checkpoints/seed{seed}_subject{subject}.pt',map_location='cpu',weights_only=True)
 assert set(checkpoint['outer_training_subjects'])==set('123456')-{subject}
 assert checkpoint['validation_subject']!=subject
 models={'unet':UNet(2),'flow':UNet(4),'diffusion':UNet(4),'cvae':ConditionalVAE(),'cgan':UNet(3),'saits':new_saits(device)}
 for name,model in models.items():model.load_state_dict(checkpoint['models'][name]);model.to(device).eval()
 waves=load_trials(ROOT/'work/physionet_bioz')[subject];examples=make_examples(waves,np.random.default_rng(42+fold),2)
 output=predict(models,examples,device,12,1000*seed+fold)
 stored=np.load(ROOT/f'work/bioz_metrics_validated_predictions/seed{seed}_subject{subject}_mask0_gap2.npz')
 for name,values in output.items():
  delta=float(np.max(np.abs(values-stored[name])))
  np.testing.assert_allclose(values,stored[name],atol=2e-5,rtol=1e-4)
  print('Checkpoint replay',name,'max absolute difference',delta)
 records=list(csv.DictReader((ROOT/'outputs/bioz_metrics_validated.sensitivity.csv').open()))
 records=[r for r in records if r['seed']==str(seed) and r['held_out_subject']==subject and r['mask_set']=='0' and r['gap_seconds']=='2']
 for name in METHODS:
  for tolerance in (.05,.1,.15):
   for prominence in (.2,.35,.5):
    stats=np.array([peak_stats(t,o,g,tolerance,prominence) for t,o,g in zip(stored['target'],stored[name],stored['mask'])]).sum(axis=0)
    r=next(r for r in records if r['method']==name and float(r['tolerance_seconds'])==tolerance and float(r['prominence'])==prominence)
    expected=np.array([float(r[k]) for k in ('timing_error_samples','matched_peaks','missed_peaks','extra_peaks')])
    np.testing.assert_array_equal(stats,expected)
 print(f'Independent checkpoint replay and all 108 detector/method combinations passed for subject {subject}, seed {seed}.')

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--subject',default='1');parser.add_argument('--seed',type=int,default=2026)
 args=parser.parse_args();audit(args.subject,args.seed)
