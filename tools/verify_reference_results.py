"""Reconcile recorded outcomes and paired intervals without raw recordings."""
from pathlib import Path
from collections import defaultdict
from statistics import mean
import csv,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'results/reference'

def read(path):
    with path.open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))

def main():
    release=json.loads((ROOT/'release_manifest.json').read_text())
    for record in release['included_files']:
        assert hashlib.sha256((ROOT/record['public']).read_bytes()).hexdigest()==record['public_sha256'],record['public']
    checked={}
    def code_hashes(record):
        for field in ('code_sha256','execution_code_sha256'):
            for name,digest in record.get(field,{}).items():
                if name not in checked:checked[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                assert checked[name]==digest,name
    original=json.loads((REF/'outputs/bioz_metrics_validated.run.json').read_text())
    sweep=json.loads((REF/'outputs/bioz_augmentation_multiseed.run.json').read_text())
    assert original['status']=='complete' and len(original['completed_folds'])==30
    assert sweep['status']=='complete' and len(sweep['cases'])==30
    code_hashes(original);code_hashes(sweep)
    assert len(original['data_sha256'])==17
    folds=read(REF/'outputs/bioz_metrics_validated.csv')
    windows=read(REF/'outputs/bioz_metrics_validated.windows.csv')
    assert len(folds)==3240 and len(windows)==183600
    def reconcile(folds,windows,keys):
        grouped=defaultdict(list)
        for row in windows:grouped[tuple(row[k] for k in keys)].append(row)
        for fold in folds:
            rows=grouped[tuple(fold[k] for k in keys)]
            assert len(rows)==int(fold['windows'])
            assert abs(mean(float(r['mae']) for r in rows)-float(fold['mae']))<1e-7
            for metric in ('matched_peaks','missed_peaks','extra_peaks','timing_error_samples'):
                assert abs(sum(float(r[metric]) for r in rows)-float(fold[metric]))<1e-6
            if 'f1' in fold:
                tp,fn,fp=(float(fold[k]) for k in ('matched_peaks','missed_peaks','extra_peaks'))
                expected=2*tp/(2*tp+fn+fp) if 2*tp+fn+fp else 0.
                assert abs(expected-float(fold['f1']))<1e-12
    reconcile(folds,windows,('seed','held_out_subject','mask_set','gap_seconds','method'))
    augmented=[];augwindows=[]
    for case in sweep['cases']:
        prefix=REF/case['prefix']
        manifest=json.loads(prefix.with_suffix('.run.json').read_text())
        assert manifest['status']=='complete' and manifest['test_scored_after_all_training']
        code_hashes(manifest)
        test=manifest['test_subject'];val=manifest['validation_subject']
        assert test not in manifest['outer_training_subjects']
        assert test not in manifest['inner_training_subjects'] and val not in manifest['inner_training_subjects']
        rows=read(prefix.with_suffix('.csv'));windowrows=read(prefix.with_suffix('.windows.csv'))
        assert len(rows)==108
        reconcile(rows,windowrows,('condition','mask_set','gap_seconds','method'))
        augmented.extend({**r,'seed':str(case['seed']),'held_out_subject':case['subject']} for r in rows)
        augwindows.extend(windowrows)
    assert len(augmented)==3240 and len(augwindows)==183600
    summary_subjects=read(REF/'outputs/bioz_augmentation_multiseed.subjects.csv')
    for row in summary_subjects:
        selected=[r for r in augmented if r['condition']==row['condition'] and r['method']==row['method']
                  and r['gap_seconds']==row['gap_seconds'] and r['held_out_subject']==row['subject']]
        assert len(selected)==15
        assert abs(mean(float(r['mae']) for r in selected)-float(row['mae']))<1e-12
        assert abs(mean(float(r['f1']) for r in selected)-float(row['macro_f1']))<1e-12
    indices=np.random.default_rng(20261118).integers(0,6,size=(10000,6))
    paired=read(REF/'outputs/bioz_augmentation_multiseed.paired.csv')
    for record in paired:
        for field,column in [('mae','mae'),('macro_f1','macro_f1')]:
            differences=[]
            for subject in '123456':
                values={r['condition']:float(r[column]) for r in summary_subjects if r['method']==record['method']
                        and r['gap_seconds']==record['gap_seconds'] and r['subject']==subject}
                differences.append(values['real_plus_synthetic']-values['real_only_matched'])
            differences=np.asarray(differences)
            low,high=np.quantile(differences[indices].mean(axis=1),(.025,.975))
            assert np.allclose([differences.mean(),low,high],[float(record[field+'_difference']),
                               float(record[field+'_bootstrap_low']),float(record[field+'_bootstrap_high'])],rtol=0,atol=1e-12)
    print('Passed: source/file hashes, 30 original folds, 30 paired cases, 367200 window outcomes, and 36 paired intervals.')

if __name__=='__main__':main()
