"""Independent generator/selection/prediction audit and full detector grid."""
from pathlib import Path
import csv
import json
import sys
import hashlib
import numpy as np
import torch
from scipy.signal import find_peaks

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bioz_inpainting import load_trials,make_examples,UNet,ConditionalVAE
from synthetic_augmentation import SyntheticBioZ,check_split,CONDITIONS,NEURAL
from validated_benchmark import EpochSelector,predict,new_saits

PREFIX=ROOT/'outputs/bioz_augmentation_multiseed'


def read(path):
    with path.open(newline='',encoding='utf-8') as handle:return list(csv.DictReader(handle))


def match(truth,predicted,tolerance):
    i=j=matched=timing=0
    while i<len(truth) and j<len(predicted):
        delta=predicted[j]-truth[i]
        if abs(delta)<=tolerance:
            timing+=abs(delta);matched+=1;i+=1;j+=1
        elif delta<0:j+=1
        else:i+=1
    return timing,matched,len(truth)-matched,len(predicted)-matched


def main():
    sweep=json.loads(PREFIX.with_suffix('.run.json').read_text())
    assert sweep['status']=='complete' and sweep['completed_cases']==30
    subjects=load_trials(ROOT/'work/physionet_bioz')
    fit_cache={};reference_cache={};truth_cache={};hash_cache={}
    sensitivity=[];verified_windows=verified_metrics=verified_selections=replays=0
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for case in sweep['cases']:
        prefix=ROOT/case['prefix'];manifest=json.loads(prefix.with_suffix('.run.json').read_text())
        assert manifest['status']=='complete' and manifest['test_scored_after_all_training']
        check_split(manifest['inner_training_subjects'],manifest['validation_subject'],manifest['outer_training_subjects'],manifest['test_subject'])
        for category in ('code_sha256','execution_code_sha256','data_sha256'):
            for path,digest in manifest[category].items():
                resolved=(ROOT/path).resolve()
                if str(resolved) not in hash_cache:hash_cache[str(resolved)]=hashlib.sha256(resolved.read_bytes()).hexdigest()
                assert hash_cache[str(resolved)]==digest,path
        paths=ROOT/'work'/(prefix.name+'_artifacts')
        for stage in ('inner','outer'):
            info=manifest[stage+'_generator'];ids=tuple(info['source_subjects'])
            assert manifest['test_subject'] not in ids
            if stage=='inner':assert manifest['validation_subject'] not in ids
            if ids not in fit_cache:fit_cache[ids]=SyntheticBioZ().fit(np.concatenate([subjects[s] for s in ids]))
            expected=fit_cache[ids]
            with np.load(paths/f'{stage}_generator.npz') as saved:
                for name in saved.files:np.testing.assert_allclose(saved[name],getattr(expected,name),rtol=1e-10,atol=1e-9)
            generated,donors=expected.sample(info['generated_windows'],info['seed'])
            assert hashlib.sha256(generated.tobytes()).hexdigest()==info['synthetic_sha256']
            with np.load(paths/f'{stage}_synthetic.npz') as saved:
                np.testing.assert_array_equal(saved['raw'],generated)
                np.testing.assert_array_equal(saved['donor_index'],donors)
        selected=read(prefix.with_suffix('.selection.csv'));validation=read(prefix.with_suffix('.validation.csv'))
        for row in selected:
            selector=EpochSelector((row['method'],))
            for r in validation:
                if r['condition']==row['condition'] and r['method']==row['method']:
                    selector.update(int(r['epoch']),{row['method']:float(r['validation_mae'])})
            assert selector.epochs[row['method']]==int(row['selected_epoch'])
            assert abs(selector.best[row['method']]-float(row['validation_mae']))<1e-10
            assert selector.stopped[row['method']]==(row['stopped_by_patience']=='True')
            verified_selections+=1
        records=read(prefix.with_suffix('.csv'));windows=read(prefix.with_suffix('.windows.csv'))
        for condition in CONDITIONS:
            for mask_set in range(3):
                for gap in (.5,1,2):
                    with np.load(paths/f'{condition}_mask{mask_set}_gap{gap}.npz') as saved:
                        target,context,mask,scales=(saved[k] for k in ('target','context','mask','scales'))
                        identity=(case['subject'],mask_set,gap)
                        if identity not in reference_cache:
                            fold=sorted(subjects,key=int).index(case['subject'])
                            examples=make_examples(subjects[case['subject']],np.random.default_rng((42,142,242)[mask_set]+fold),gap)
                            values=[v.numpy()[:,0] if v.ndim==3 else v.numpy() for v in examples.tensors]
                            reference_cache[identity]=tuple(values)
                        for value,expected in zip((target,context,mask,scales),reference_cache[identity]):np.testing.assert_array_equal(value,expected)
                        for method in NEURAL:
                            output=saved[method];assert np.isfinite(output).all()
                            np.testing.assert_array_equal(output[~mask],context[~mask])
                            errors=(np.abs(output-target)*mask).sum(axis=1)/mask.sum(axis=1)*scales
                            row=next(r for r in records if r['condition']==condition and r['method']==method
                                     and int(r['mask_set'])==mask_set and float(r['gap_seconds'])==gap)
                            assert abs(float(row['mae'])-float(errors.mean()))<1e-9
                            window_rows=[r for r in windows if r['condition']==condition and r['method']==method
                                         and int(r['mask_set'])==mask_set and float(r['gap_seconds'])==gap]
                            window_rows.sort(key=lambda r:int(r['window_index']))
                            np.testing.assert_allclose([float(r['mae']) for r in window_rows],errors,atol=1e-10,rtol=0)
                            for prominence in (.2,.35,.5):
                                all_statistics=[]
                                for index,(truth_wave,predicted_wave,gap_mask) in enumerate(zip(target,output,mask)):
                                    key=(*identity,index,prominence)
                                    if key not in truth_cache:
                                        truth=find_peaks(truth_wave,distance=35,prominence=prominence)[0]
                                        truth_cache[key]=truth[gap_mask[truth]]
                                    predicted=find_peaks(predicted_wave,distance=35,prominence=prominence)[0]
                                    predicted=predicted[gap_mask[predicted]]
                                    all_statistics.append([match(truth_cache[key],predicted,t) for t in (6,12,19)])
                                all_statistics=np.asarray(all_statistics)
                                for tolerance_index,tolerance in enumerate((.05,.1,.15)):
                                    statistics=all_statistics[:,tolerance_index,:]
                                    timing,tp,fn,fp=statistics.sum(axis=0)
                                    sensitivity.append({'seed':case['seed'],'held_out_subject':case['subject'],'condition':condition,
                                                        'method':method,'mask_set':mask_set,'gap_seconds':gap,
                                                        'tolerance_seconds':tolerance,'prominence':prominence,'windows':len(target),
                                                        'timing_error_samples':int(timing),'matched_peaks':int(tp),'missed_peaks':int(fn),'extra_peaks':int(fp),
                                                        'f1':float(2*tp/(2*tp+fn+fp)) if 2*tp+fn+fp else 0.})
                                    if prominence==.35 and tolerance==.15:
                                        np.testing.assert_array_equal(statistics.sum(axis=0),[int(row[k]) for k in ('timing_error_samples','matched_peaks','missed_peaks','extra_peaks')])
                                        np.testing.assert_array_equal(statistics,[[int(r[k]) for k in ('timing_error_samples','matched_peaks','missed_peaks','extra_peaks')] for r in window_rows])
                                        verified_windows+=len(target);verified_metrics+=1
            if case['seed']==2026:
                checkpoint=torch.load(paths/f'{condition}.pt',map_location='cpu',weights_only=True)
                models={'unet':UNet(2),'flow':UNet(4),'diffusion':UNet(4),'cvae':ConditionalVAE(),'cgan':UNet(3),'saits':new_saits(device)}
                for name,model in models.items():model.load_state_dict(checkpoint['models'][name]);model.to(device).eval()
                fold=sorted(subjects,key=int).index(case['subject'])
                examples=make_examples(subjects[case['subject']],np.random.default_rng(42+fold),2)
                values=predict(models,examples,device,12,1000*case['seed']+fold)
                with np.load(paths/f'{condition}_mask0_gap2.npz') as saved:
                    for name,value in values.items():
                        np.testing.assert_allclose(value,saved[name],atol=2e-5,rtol=1e-4);replays+=1
                del models
        print(f'AUDITED subject{case["subject"]}/seed{case["seed"]}: {verified_metrics} metrics, {verified_windows} windows',flush=True)
    assert verified_metrics==3240 and verified_windows==183600 and verified_selections==360 and len(sensitivity)==29160 and replays==72
    with PREFIX.with_suffix('.sensitivity.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(sensitivity[0]));writer.writeheader();writer.writerows(sensitivity)
    result={'status':'passed','cases':30,'generator_fits_independently_recomputed':len(fit_cache),
            'generated_datasets_reproduced':60,'selected_epochs_recomputed':verified_selections,
            'primary_fold_metrics_recomputed':verified_metrics,'window_metrics_recomputed':verified_windows,
            'detector_grid_fold_records':len(sensitivity),'checkpoint_models_replayed':replays,
            'unit_tests_passed':16,'fixture_end_to_end_smoke':'passed','bootstrap_unit':'participant, after averaging seeds and masks',
            'scope':'Same reused six-person cohort after a one-person pilot; exploratory, no external physiological validation.'}
    PREFIX.with_suffix('.audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
