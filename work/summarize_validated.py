"""Audit and summarize the validated 12-method benchmark, keeping subjects distinct."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
from statistics import mean, stdev, median
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/bioz_metrics_validated.csv'
METHODS=('linear','prefill','pchip','periodic','gp_mean','unet','saits','cvae','cgan','flow','diffusion','gp')
NEURAL=('unet','saits','cvae','cgan','flow','diffusion')
GAPS=('0.5','1','2');SEEDS=tuple(range(2026,2031));SUBJECTS=tuple(str(i) for i in range(1,7))
MASKS=('0','1','2');COUNTS=dict(zip(SUBJECTS,(40,40,40,60,80,80)))


def read(suffix):
 return list(csv.DictReader(SOURCE.with_suffix(suffix).open(newline='',encoding='utf-8')))


def write(suffix,records):
 with SOURCE.with_suffix(suffix).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def peaks(rows):
 tp,fn,fp=(sum(float(r[k]) for r in rows) for k in ('matched_peaks','missed_peaks','extra_peaks'))
 timing=sum(float(r['timing_error_samples']) for r in rows)*1000/125/tp if tp else None
 return {'f1':2*tp/(2*tp+fn+fp) if 2*tp+fn+fp else 0.,
         'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,
         'matched_peaks':tp,'missed_peaks':fn,'extra_peaks':fp,'timing_mae_ms':timing}


def summarize():
 manifest=json.loads(SOURCE.with_suffix('.run.json').read_text())
 assert manifest['status']=='complete' and len(manifest['completed_folds'])==30
 for field in ('code_sha256','data_sha256'):
  for path,digest in manifest[field].items(): assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
 for fold in manifest['folds']:
  test=fold['test_subject'];val=fold['validation_subject'];inner=fold['inner_training_subjects'];outer=fold['outer_training_subjects']
  assert test!=val and test not in inner and val not in inner and test not in outer
  assert len(inner)==4 and set(inner+[val])==set(outer) and len(outer)==5
 rows=read('.csv'); assert len(rows)==5*6*3*3*12
 lookup={(int(r['seed']),r['held_out_subject'],r['mask_set'],r['gap_seconds'],r['method']):r for r in rows}
 assert len(lookup)==len(rows)
 for r in rows:assert int(r['windows'])==COUNTS[r['held_out_subject']]
 summaries=[];subject_rows=[];seed_rows=[];mask_rows=[]
 for gap in GAPS:
  for method in METHODS:
   participant_maes=[]
   for subject in SUBJECTS:
    a=[lookup[seed,subject,mask,gap,method] for seed in SEEDS for mask in MASKS]
    val=mean(float(r['mae']) for r in a);participant_maes.append(val)
    subject_rows.append({'subject':subject,'gap_seconds':gap,'method':method,'mae':val,
                         'f1_macro_component':mean(peaks([r])['f1'] for r in a)})
   run_records=[]
   for seed in SEEDS:
    placements=[]
    for mask in MASKS:
     a=[lookup[seed,subject,mask,gap,method] for subject in SUBJECTS]
     record={'seed':seed,'mask_set':mask,'gap_seconds':gap,'method':method,
             'mae':mean(float(r['mae']) for r in a),**peaks(a),
             'macro_f1':mean(peaks([r])['f1'] for r in a)}
     placements.append(record);mask_rows.append(record)
    record={'seed':seed,'gap_seconds':gap,'method':method}
    for key in ('mae','f1','macro_f1','precision','recall','matched_peaks','missed_peaks','extra_peaks','timing_mae_ms'):
     values=[r[key] for r in placements if r[key] is not None]
     record[key]=mean(values) if values else None
    run_records.append(record);seed_rows.append(record)
   record={'method':method,'gap_seconds':gap,'mae':mean(participant_maes),'mae_subject_sd':stdev(participant_maes)}
   for key in ('mae','f1','macro_f1','precision','recall','matched_peaks','missed_peaks','extra_peaks','timing_mae_ms'):
    values=[r[key] for r in run_records if r[key] is not None]
    record[key]=mean(values) if values else None;record[key+'_seed_sd']=stdev(values) if values else None
   summaries.append(record)
 # Audits of shared reference peaks, mask positions, fold-window reconciliation.
 windows=read('.windows.csv');assert len(windows)==5*340*3*3*12
 window_groups={};gaps={};truth={}
 for r in windows:
  key=(int(r['seed']),r['held_out_subject'],r['mask_set'],r['gap_seconds'],r['method'])
  window_groups.setdefault(key,[]).append(r)
  identity=(r['held_out_subject'],r['mask_set'],r['gap_seconds'],r['window_index'])
  gaps.setdefault(identity,r['gap_start']);assert gaps[identity]==r['gap_start']
  count=float(r['matched_peaks'])+float(r['missed_peaks']);truth.setdefault(identity,count);assert truth[identity]==count
 for key,a in window_groups.items():
  b=lookup[key];assert len(a)==int(b['windows'])
  assert abs(mean(float(r['mae']) for r in a)-float(b['mae']))<1e-7
  for k in ('timing_error_samples','matched_peaks','missed_peaks','extra_peaks'):
   assert abs(sum(float(r[k]) for r in a)-float(b[k]))<1e-6
 sensitivity=read('.sensitivity.csv');assert len(sensitivity)==5*6*3*3*12*9
 # Matching tolerance changes associations, not the detected peak populations.
 monotonic={}
 for r in sensitivity:
  key=tuple(r[k] for k in ('seed','held_out_subject','mask_set','gap_seconds','method','prominence'))
  monotonic.setdefault(key,[]).append(r)
 for a in monotonic.values():
  a.sort(key=lambda r:float(r['tolerance_seconds']))
  assert len(a)==3
  assert len({float(r['matched_peaks'])+float(r['missed_peaks']) for r in a})==1
  assert len({float(r['matched_peaks'])+float(r['extra_peaks']) for r in a})==1
  matches=[float(r['matched_peaks']) for r in a];assert matches==sorted(matches)
 groups={}
 for r in sensitivity:
  key=(r['mask_set'],r['gap_seconds'],r['method'],r['tolerance_seconds'],r['prominence'],int(r['seed']))
  groups.setdefault(key,[]).append(r)
 settings={}
 for (mask,gap,method,tolerance,prominence,seed),a in groups.items():
  k=(mask,gap,method,tolerance,prominence)
  settings.setdefault(k,[]).append({**peaks(a),'macro_f1':mean(peaks([r])['f1'] for r in a)})
 sensitivity_summary=[]
 for (mask,gap,method,tolerance,prominence),a in settings.items():
  sensitivity_summary.append({'mask_set':mask,'gap_seconds':gap,'method':method,'tolerance_seconds':tolerance,
                               'prominence':prominence,'f1':mean(r['f1'] for r in a),'f1_seed_sd':stdev(r['f1'] for r in a),
                               'macro_f1':mean(r['macro_f1'] for r in a)})
 # Each of the 27 settings is retained; no detector or mask is selected using test scores.
 sensitivity_lookup={(r['mask_set'],r['gap_seconds'],r['tolerance_seconds'],r['prominence'],r['method']):r for r in sensitivity_summary}
 robustness=[]
 for gap in GAPS:
  combinations=list(itertools.product(MASKS,('0.05','0.1','0.15'),('0.2','0.35','0.5')))
  for method in METHODS:
   values=[];ranks=[];macro_ranks=[]
   for mask,tolerance,prominence in combinations:
    candidates=[sensitivity_lookup[mask,gap,tolerance,prominence,m] for m in METHODS]
    best=sorted(candidates,key=lambda r:r['f1'],reverse=True)
    best_macro=sorted(candidates,key=lambda r:r['macro_f1'],reverse=True)
    ranks.append(1+sum(r['f1']>sensitivity_lookup[mask,gap,tolerance,prominence,method]['f1']+1e-12 for r in candidates))
    macro_ranks.append(1+sum(r['macro_f1']>sensitivity_lookup[mask,gap,tolerance,prominence,method]['macro_f1']+1e-12 for r in candidates))
    values.append(sensitivity_lookup[mask,gap,tolerance,prominence,method]['f1'])
   robustness.append({'gap_seconds':gap,'method':method,'f1_min':min(values),'f1_max':max(values),
                      'rank_min':min(ranks),'rank_max':max(ranks),'first_place_settings':sum(rank==1 for rank in ranks),
                      'macro_first_place_settings':sum(rank==1 for rank in macro_ranks),'settings':len(combinations)})
 selections=read('.selection.csv');assert len(selections)==30*6
 selection_summary=[]
 for method in NEURAL:
  a=[r for r in selections if r['method']==method];epochs=[int(r['selected_epoch']) for r in a]
  selection_summary.append({'method':method,'median_epoch':median(epochs),'min_epoch':min(epochs),'max_epoch':max(epochs),
                            'at_epoch_cap':sum(e==60 for e in epochs),'budget_limited_runs':sum(int(r['stopped_by_patience'])==0 for r in a),'patience_stops':sum(int(r['stopped_by_patience']) for r in a),'runs':len(a)})
 # Exploratory paired subject bootstrap after averaging seeds and mask schedules.
 # These descriptive intervals do not make the overlapping LOSO fits independent.
 rng=np.random.default_rng(20261118);indices=rng.integers(0,6,size=(10000,6))
 paired=[]
 subject_lookup={(r['subject'],r['gap_seconds'],r['method']):r for r in subject_rows}
 for gap in GAPS:
  for method in METHODS:
   if method=='unet':continue
   diff=np.array([subject_lookup[s,gap,method]['mae']-subject_lookup[s,gap,'unet']['mae'] for s in SUBJECTS])
   low,high=np.quantile(diff[indices].mean(axis=1),(.025,.975))
   paired.append({'gap_seconds':gap,'method_minus_unet':method,'mean_mae_difference':float(diff.mean()),
                  'descriptive_subject_bootstrap_2.5':float(low),'descriptive_subject_bootstrap_97.5':float(high)})
 for suffix,data in (('.summary.csv',summaries),('.subjects.csv',subject_rows),('.seeds.csv',seed_rows),
                     ('.masks.csv',mask_rows),('.sensitivity_summary.csv',sensitivity_summary),('.robustness.csv',robustness),
                     ('.selection_summary.csv',selection_summary),('.paired.csv',paired)):
  write(suffix,data)
 print('Validated 30 outer fits, 180 selected epochs, 3,240 fold metrics, 29,160 sensitivity records, 183,600 window records, shared masks, disjoint splits, and source/data hashes.')
 for r in summaries:
  if r['gap_seconds']=='2':print(f"{r['method']:10s} MAE {r['mae']:.3f} F1 {r['f1']:.3f} macroF1 {r['macro_f1']:.3f}")
 return summaries,subject_rows,seed_rows,robustness,selection_summary

if __name__=='__main__':summarize()
