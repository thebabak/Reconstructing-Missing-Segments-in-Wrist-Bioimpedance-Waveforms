"""Participant-aware summaries of all completed paired augmentation cases."""
from pathlib import Path
import csv
import hashlib
import itertools
import json
from statistics import mean, stdev, median
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PREFIX=ROOT/'outputs/bioz_augmentation_multiseed'
CONDITIONS=('real_only_matched','real_plus_synthetic')
METHODS=('unet','saits','cvae','cgan','flow','diffusion')
GAPS=('0.5','1','2');SEEDS=tuple(range(2026,2031));SUBJECTS=tuple('123456');MASKS=('0','1','2')
COUNTS=dict(zip(SUBJECTS,(40,40,40,60,80,80)))


def read(path):
    with path.open(newline='',encoding='utf-8') as handle:return list(csv.DictReader(handle))


def write(suffix,rows):
    with PREFIX.with_suffix(suffix).open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def peak_summary(rows):
    tp,fn,fp=(sum(float(r[k]) for r in rows) for k in ('matched_peaks','missed_peaks','extra_peaks'))
    return {'f1':2*tp/(2*tp+fn+fp) if 2*tp+fn+fp else 0.,
            'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,
            'macro_f1':mean(float(r['f1']) for r in rows),
            'timing_mae_ms':sum(float(r['timing_error_samples']) for r in rows)*8/tp if tp else None}


def load_completed():
    sweep=json.loads(PREFIX.with_suffix('.run.json').read_text())
    assert sweep['status']=='complete' and sweep['completed_cases']==30 and len(sweep['cases'])==30
    assert set(sweep['subjects'])==set(SUBJECTS) and tuple(sweep['seeds'])==SEEDS
    assert {(r['subject'],r['seed']) for r in sweep['cases']}==set(itertools.product(SUBJECTS,SEEDS))
    checked={}
    def verify(hashes):
        for path,digest in hashes.items():
            resolved=(ROOT/path).resolve()
            if str(resolved) not in checked:checked[str(resolved)]=hashlib.sha256(resolved.read_bytes()).hexdigest()
            assert checked[str(resolved)]==digest,path
    verify(sweep['code_sha256'])
    folds=[];windows=[];selections=[]
    for case in sweep['cases']:
        prefix=ROOT/case['prefix'];manifest=json.loads(prefix.with_suffix('.run.json').read_text())
        assert case['status']=='complete' and manifest['status']=='complete' and manifest['test_scored_after_all_training']
        assert manifest['test_subject']==case['subject'] and int(manifest['configuration']['seed'])==case['seed']
        test=manifest['test_subject'];validation=manifest['validation_subject']
        inner=manifest['inner_training_subjects'];outer=manifest['outer_training_subjects']
        assert test!=validation and test not in outer and validation not in inner and test not in inner
        assert len(inner)==4 and len(outer)==5 and set(inner+[validation])==set(outer)
        assert manifest['inner_generator']['source_subjects']==inner and manifest['outer_generator']['source_subjects']==outer
        assert manifest['inner_generator']['generated_windows']==sum(COUNTS[s] for s in inner)
        assert manifest['outer_generator']['generated_windows']==sum(COUNTS[s] for s in outer)
        for name in ('code_sha256','execution_code_sha256','data_sha256'):verify(manifest[name])
        metadata={'seed':case['seed'],'held_out_subject':case['subject']}
        a=read(prefix.with_suffix('.csv'));assert len(a)==108
        for row in a:
            assert int(row['windows'])==COUNTS[test]
            assert abs(float(row['f1'])-peak_summary([row])['f1'])<1e-12
            folds.append({**metadata,**row})
        a=read(prefix.with_suffix('.windows.csv'));assert len(a)==COUNTS[test]*108
        windows.extend({**metadata,**row} for row in a)
        a=read(prefix.with_suffix('.selection.csv'));assert len(a)==12
        selections.extend({**metadata,'validation_subject':validation,**row} for row in a)
    assert len(folds)==3240 and len(windows)==183600 and len(selections)==360
    lookup={(str(r['seed']),r['held_out_subject'],r['condition'],r['method'],r['gap_seconds'],r['mask_set']):r for r in folds}
    assert len(lookup)==3240
    # Mask/reference counts must agree across conditions, methods and seeds.
    truth={}
    groups={}
    for row in windows:
        identity=tuple(str(row[k]) for k in ('held_out_subject','gap_seconds','mask_set','window_index'))
        count=int(row['matched_peaks'])+int(row['missed_peaks'])
        if identity in truth:assert truth[identity]==count
        truth[identity]=count
        key=tuple(str(row[k]) for k in ('seed','held_out_subject','condition','method','gap_seconds','mask_set'))
        groups.setdefault(key,[]).append(row)
    for key,group in groups.items():
        row=lookup[key];assert len(group)==int(row['windows'])
        assert abs(mean(float(r['mae']) for r in group)-float(row['mae']))<1e-7
        for metric in ('matched_peaks','missed_peaks','extra_peaks','timing_error_samples'):
            assert sum(int(r[metric]) for r in group)==int(row[metric])
    return sweep,folds,windows,selections


def summarize():
    sweep,folds,windows,selections=load_completed()
    subjects=[];seeds=[];summary=[]
    for condition,method,gap in itertools.product(CONDITIONS,METHODS,GAPS):
        group=[r for r in folds if r['condition']==condition and r['method']==method and r['gap_seconds']==gap]
        participant=[];runs=[]
        for subject in SUBJECTS:
            a=[r for r in group if r['held_out_subject']==subject];assert len(a)==15
            record={'condition':condition,'method':method,'gap_seconds':gap,'subject':subject,
                    'mae':mean(float(r['mae']) for r in a),'macro_f1':mean(float(r['f1']) for r in a)}
            subjects.append(record);participant.append(record)
        for seed in SEEDS:
            placements=[]
            for mask in MASKS:
                a=[r for r in group if r['seed']==seed and r['mask_set']==mask];assert len(a)==6
                placements.append({'mae':mean(float(r['mae']) for r in a),**peak_summary(a)})
            record={'condition':condition,'method':method,'gap_seconds':gap,'seed':seed}
            for metric in placements[0]:
                values=[r[metric] for r in placements if r[metric] is not None]
                record[metric]=mean(values) if values else None
            seeds.append(record);runs.append(record)
        record={'condition':condition,'method':method,'gap_seconds':gap,
                'mae':mean(r['mae'] for r in participant),'mae_subject_sd':stdev(r['mae'] for r in participant)}
        for metric in ('mae','f1','macro_f1','precision','recall','timing_mae_ms'):
            values=[r[metric] for r in runs if r[metric] is not None]
            record[metric]=mean(values) if values else None
            record[metric+'_seed_sd']=stdev(values) if values else None
        summary.append(record)
    participant={(r['condition'],r['method'],r['gap_seconds'],r['subject']):r for r in subjects}
    aggregate={(r['condition'],r['method'],r['gap_seconds']):r for r in summary}
    indices=np.random.default_rng(20261118).integers(0,6,size=(10000,6))
    paired=[]
    for method,gap in itertools.product(METHODS,GAPS):
        record={'method':method,'gap_seconds':gap}
        for metric in ('mae','macro_f1'):
            delta=np.array([participant['real_plus_synthetic',method,gap,s][metric]-participant['real_only_matched',method,gap,s][metric] for s in SUBJECTS])
            low,high=np.quantile(delta[indices].mean(axis=1),(.025,.975))
            record.update({metric+'_difference':float(delta.mean()),metric+'_bootstrap_low':float(low),metric+'_bootstrap_high':float(high),
                           metric+'_participants_improved':sum(delta<0) if metric=='mae' else sum(delta>0)})
        record['pooled_f1_difference']=aggregate['real_plus_synthetic',method,gap]['f1']-aggregate['real_only_matched',method,gap]['f1']
        record={k:int(v) if isinstance(v,(np.integer,)) else v for k,v in record.items()}
        paired.append(record)
    selection_summary=[]
    for condition,method in itertools.product(CONDITIONS,METHODS):
        a=[r for r in selections if r['condition']==condition and r['method']==method];assert len(a)==30
        epochs=[int(r['selected_epoch']) for r in a]
        selection_summary.append({'condition':condition,'method':method,'runs':30,'median_epoch':median(epochs),
                                  'min_epoch':min(epochs),'max_epoch':max(epochs),'selected_at_cap':sum(e==120 for e in epochs),
                                  'patience_stops':sum(r['stopped_by_patience']=='True' for r in a),
                                  'budget_limited':sum(r['stopped_by_patience']=='False' for r in a)})
    for suffix,rows in (('.csv',folds),('.windows.csv',windows),('.selection.csv',selections),('.subjects.csv',subjects),
                        ('.seeds.csv',seeds),('.summary.csv',summary),('.paired.csv',paired),('.selection_summary.csv',selection_summary)):
        write(suffix,rows)
    sensitivity_path=PREFIX.with_suffix('.sensitivity.csv')
    effects=[]
    if sensitivity_path.exists():
        sensitivity=read(sensitivity_path);assert len(sensitivity)==29160
        settings={}
        for condition,method,gap,mask,tolerance,prominence in itertools.product(CONDITIONS,METHODS,GAPS,MASKS,('0.05','0.1','0.15'),('0.2','0.35','0.5')):
            a=[r for r in sensitivity if r['condition']==condition and r['method']==method and r['gap_seconds']==gap
               and r['mask_set']==mask and r['tolerance_seconds']==tolerance and r['prominence']==prominence]
            assert len(a)==30
            runs=[peak_summary([r for r in a if int(r['seed'])==seed]) for seed in SEEDS]
            settings[condition,method,gap,mask,tolerance,prominence]={'f1':mean(r['f1'] for r in runs),'macro_f1':mean(r['macro_f1'] for r in runs)}
        for method,gap in itertools.product(METHODS,GAPS):
            differences=[];macros=[]
            for mask,tolerance,prominence in itertools.product(MASKS,('0.05','0.1','0.15'),('0.2','0.35','0.5')):
                real,mixed=[settings[c,method,gap,mask,tolerance,prominence] for c in CONDITIONS]
                differences.append(mixed['f1']-real['f1']);macros.append(mixed['macro_f1']-real['macro_f1'])
            effects.append({'method':method,'gap_seconds':gap,'settings':27,'pooled_delta_min':min(differences),
                            'pooled_delta_max':max(differences),'pooled_positive_settings':sum(v>0 for v in differences),
                            'macro_delta_min':min(macros),'macro_delta_max':max(macros),'macro_positive_settings':sum(v>0 for v in macros)})
        write('.sensitivity_effects.csv',effects)
    result={'status':'aggregated','cases':30,'conditions':2,'neural_methods':6,'refitted_neural_models':360,
            'fold_records':len(folds),'window_records':len(windows),'selection_records':len(selections),
            'sensitivity_records':29160 if effects else 0,'bootstrap_replicates':10000,'participants':6,
            'interpretation':'Paired participant bootstrap after averaging five runs and three masks. Exploratory, conditional on fitted models, no multiplicity correction; same reused cohort, not external validation.'}
    PREFIX.with_suffix('.analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    for r in paired:
        if r['gap_seconds']=='2':print(r)
    return summary,paired,selection_summary,effects


if __name__=='__main__':summarize()
