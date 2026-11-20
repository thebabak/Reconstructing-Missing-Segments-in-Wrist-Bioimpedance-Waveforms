"""Subject-held-out validation, additional baselines, and peak-sensitivity benchmark."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from torch.utils.data import DataLoader

from bioz_inpainting import (
    RATE, WINDOW, PADDING, DECIMATION, GAPS, UNet, ConditionalVAE,
    GaussianProcess, load_trials, make_examples, preprocess, peak_stats, smoke_data, train,
)
from vendor.saits.saits import SAITS

CNN_NAMES = ('unet', 'flow', 'diffusion', 'cvae', 'cgan')
METHODS = ('linear', 'prefill', 'pchip', 'periodic', 'gp_mean', 'unet', 'saits', 'cvae', 'cgan', 'flow', 'diffusion', 'gp')
TOLERANCES = (0.05, 0.10, 0.15)
PROMINENCES = (0.20, 0.35, 0.50)
MASK_SEEDS = (42, 142, 242)


def subject_split(subjects, held_out, seed):
    available = sorted(set(subjects) - {held_out}, key=int)
    validation = available[(seed - 2026) % len(available)]
    return [s for s in available if s != validation], validation, available


def new_saits(device):
    return SAITS(1, 1, WINDOW, 1, 32, 64, 2, 16, 16, 0.1,
                 input_with_mask=True, param_sharing_strategy='between_group',
                 MIT=True, device=str(device), diagonal_attention_mask=True).to(device)


@torch.inference_mode()
def predict(models, examples, device, steps, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    outputs = {name: [] for name in models}
    for target, context, mask, _ in DataLoader(examples, batch_size=16):
        context, mask = context.to(device), mask.to(device)
        condition = torch.cat((context, mask), dim=1)
        noise = torch.randn(context.shape, device=device, generator=generator)
        batch = {}
        if 'unet' in models: batch['unet'] = models['unet'](condition)
        if 'cgan' in models: batch['cgan'] = models['cgan'](torch.cat((condition, noise), dim=1))
        if 'cvae' in models:
            latent = torch.randn((len(context), 16), device=device, generator=generator)
            batch['cvae'] = models['cvae'].sample(condition, latent)
        state = torch.where(mask, noise, context)
        diffused = state.clone()
        for step in range(steps):
            if 'flow' in models:
                velocity = models['flow'](torch.cat((condition, state, torch.full_like(state, step/steps)), dim=1))
                state = torch.where(mask, state + velocity/steps, context)
            if 'diffusion' in models:
                velocity = models['diffusion'](torch.cat((condition, diffused, torch.full_like(diffused, 1-step/steps)), dim=1))
                delta = np.pi/(2*steps)
                diffused = torch.where(mask, np.cos(delta)*diffused - np.sin(delta)*velocity, context)
        if 'flow' in models: batch['flow'] = state
        if 'diffusion' in models: batch['diffusion'] = diffused
        if 'saits' in models:
            inputs = {'X': context.transpose(1,2), 'missing_mask': (~mask).float().transpose(1,2)}
            batch['saits'] = models['saits'].impute(inputs)[0].transpose(1,2)
        for name, proposal in batch.items():
            outputs[name].append(torch.where(mask, proposal, context)[:,0].cpu().numpy())
    return {name: np.concatenate(values) for name, values in outputs.items()}


def validation_scores(models, examples_by_gap, device, steps, seed, draws=2):
    previous = {name: model.training for name, model in models.items()}
    for model in models.values(): model.eval()
    scores = {name: [] for name in models}
    # Validation may consume DataLoader bookkeeping RNG, but must not alter training randomness.
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        for examples in examples_by_gap:
            target, _, mask, scales = [value.numpy()[:,0] if value.ndim == 3 else value.numpy() for value in examples.tensors]
            for draw in range(draws):
                outputs = predict(models, examples, device, steps, seed+10000*draw)
                for name, output in outputs.items():
                    errors = (np.abs(output-target)*mask).sum(axis=1)/mask.sum(axis=1)*scales
                    scores[name].append(float(errors.mean()))
    for name, model in models.items(): model.train(previous[name])
    return {name: float(np.mean(values)) for name, values in scores.items()}


class EpochSelector:
    def __init__(self, names, patience=3):
        self.best = {name: float('inf') for name in names}
        self.epochs = {name: 0 for name in names}
        self.wait = {name: 0 for name in names}
        self.stopped = {name: False for name in names}
        self.patience = patience

    def update(self, epoch, scores):
        for name, score in scores.items():
            if self.stopped[name]: continue
            if not np.isfinite(score): raise FloatingPointError(f'Invalid validation score: {name}')
            if score < self.best[name] - 1e-6:
                self.best[name] = score; self.epochs[name] = epoch; self.wait[name] = 0
            else:
                self.wait[name] += 1
                if self.wait[name] >= self.patience: self.stopped[name] = True
        return all(self.stopped.values())


def train_saits(waves, epochs, device, seed, callback, log):
    torch.manual_seed(seed)
    model = new_saits(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs+1):
        model.train(); loss_sum = 0.; count = 0
        for target, context, mask, _ in DataLoader(make_examples(waves, rng), batch_size=16, shuffle=True):
            target, context, mask = target.to(device), context.to(device), mask.to(device)
            inputs = {'X': context.transpose(1,2), 'missing_mask': (~mask).float().transpose(1,2),
                      'X_holdout': target.transpose(1,2), 'indicating_mask': mask.float().transpose(1,2)}
            output = model(inputs, 'train')
            loss = output['reconstruction_loss'] + output['imputation_loss']
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += float(loss.detach())*len(target); count += len(target)
        value = loss_sum/count
        if not np.isfinite(value): raise FloatingPointError('Nonfinite SAITS objective')
        log(epoch, value)
        if callback(epoch, model): break
    return model.eval()


def periodic_fill(context, gap):
    """Two-harmonic least-squares extrapolation; frequency fit uses visible data only."""
    t = np.arange(WINDOW)/RATE
    best_error = np.inf; best = None
    for frequency in np.arange(0.7, 3.0001, 0.025):
        angle = 2*np.pi*frequency*t
        design = np.column_stack((np.ones(WINDOW), np.sin(angle), np.cos(angle), np.sin(2*angle), np.cos(2*angle)))
        visible = design[~gap]
        beta = np.linalg.solve(visible.T@visible + 1e-4*np.eye(5), visible.T@context[~gap])
        error = np.square(visible@beta - context[~gap]).mean()
        if error < best_error: best_error, best = error, design@beta
    output = context.copy(); output[gap] = best[gap]
    return output


def prefill_wave(raw, gap):
    positions = np.flatnonzero(gap)
    a=(PADDING+positions[0])*DECIMATION; b=(PADDING+positions[-1]+1)*DECIMATION
    filled=raw.copy()
    filled[a:b]=np.linspace(raw[a-1],raw[b],b-a+2)[1:-1]
    wave=preprocess(filled)
    return (wave-wave[~gap].mean())/max(float(wave[~gap].std()),1e-6)


def classical_predictions(raws, examples, gp):
    _, contexts, masks, _ = examples.tensors
    contexts, masks = contexts[:,0].numpy(), masks[:,0].numpy()
    outputs={m:[] for m in ('linear','prefill','pchip','periodic','gp_mean')}
    index=np.arange(WINDOW)
    for raw,context,gap in zip(raws,contexts,masks):
        outputs['linear'].append(np.interp(index,index[~gap],context[~gap]))
        outputs['prefill'].append(prefill_wave(raw,gap))
        pchip=context.copy(); pchip[gap]=PchipInterpolator(index[~gap],context[~gap])(index[gap])
        outputs['pchip'].append(pchip)
        outputs['periodic'].append(periodic_fill(context,gap))
        outputs['gp_mean'].append(gp.mean(context,gap))
    return {name:np.stack(values) for name,values in outputs.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-epochs',type=int,default=60)
    parser.add_argument('--check-every',type=int,default=5)
    parser.add_argument('--steps',type=int,default=12)
    parser.add_argument('--seeds',type=int,nargs='+',default=[2026,2027,2028,2029,2030])
    parser.add_argument('--smoke-test',action='store_true')
    args=parser.parse_args()
    if args.smoke_test == (args.data is not None): parser.error('Choose --data or --smoke-test')
    if min(args.max_epochs,args.check_every,args.steps)<1: parser.error('Positive budgets required')
    if len(set(args.seeds))!=len(args.seeds): parser.error('Distinct seeds required')
    subjects=smoke_data() if args.smoke_test else load_trials(args.data)
    if len(subjects)<3: parser.error('At least three subjects required')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    checkpoint_dir=args.output.parent/(args.output.stem+'_checkpoints');checkpoint_dir.mkdir(exist_ok=True)
    prediction_dir=Path('work')/(args.output.stem+'_predictions');prediction_dir.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),Path('bioz_inpainting.py'),*Path('vendor/saits').glob('*.py'),Path('vendor/saits/provenance.json')]
    manifest={'status':'running','configuration':vars(args).copy(),'methods':METHODS,'masks':MASK_SEEDS,
              'tolerances':TOLERANCES,'prominences':PROMINENCES,'completed_folds':[],
              'validation':'One rotating training-only subject; two fixed draws; macro mean over three durations; checkpoint patience 3 checks; refit all five outer training subjects at method-specific selected epoch.',
              'code_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              'data_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.data.rglob('*_bioz.csv')} if args.data else {},
              'windows_per_subject':{k:len(v) for k,v in subjects.items()},'device':str(device),
              'versions':{'torch':torch.__version__,'numpy':np.__version__},'folds':[]}
    manifest['configuration']={k:str(v) if isinstance(v,Path) else v for k,v in manifest['configuration'].items()}
    manifest_path=args.output.with_suffix('.run.json')
    def save_manifest(): manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    save_manifest();started=time.perf_counter()
    streams=[]
    def writer(suffix,header):
        stream=args.output.with_suffix(suffix).open('w',newline='',encoding='utf-8');streams.append(stream)
        w=csv.writer(stream);w.writerow(header);return w
    base=['seed','held_out_subject','mask_set','gap_seconds','method']
    metrics=writer('.csv',base+['windows','mae','timing_error_samples','matched_peaks','missed_peaks','extra_peaks'])
    sensitivities=writer('.sensitivity.csv',base+['tolerance_seconds','prominence','windows','timing_error_samples','matched_peaks','missed_peaks','extra_peaks'])
    windows=writer('.windows.csv',base+['window_index','gap_start','mae','timing_error_samples','matched_peaks','missed_peaks','extra_peaks'])
    validation_log=writer('.validation.csv',['seed','held_out_subject','validation_subject','epoch','method','validation_mae','stopped'])
    training_log=writer('.training.csv',['seed','held_out_subject','stage','epoch','method','objective'])
    selections=writer('.selection.csv',['seed','held_out_subject','validation_subject','method','selected_epoch','validation_mae','stopped_by_patience'])
    for fold,held_out in enumerate(sorted(subjects,key=int)):
        outer_ids=[s for s in subjects if s!=held_out]
        outer=np.concatenate([subjects[s] for s in outer_ids])
        gp=GaussianProcess(np.stack([preprocess(raw) for raw in outer]))
        test_cache={}
        for mask_set,mask_seed in enumerate(MASK_SEEDS):
            for seconds in GAPS:
                examples=make_examples(subjects[held_out],np.random.default_rng(mask_seed+fold),seconds)
                test_cache[mask_set,seconds]=(examples,classical_predictions(subjects[held_out],examples,gp))
        for run_seed in args.seeds:
            inner_ids,val_id,outer_ids=subject_split(subjects,held_out,run_seed)
            assert held_out not in inner_ids and val_id not in inner_ids and held_out!=val_id
            inner=np.concatenate([subjects[s] for s in inner_ids])
            val_examples=[make_examples(subjects[val_id],np.random.default_rng(31415+fold),g) for g in GAPS]
            training_seed=1000*run_seed+fold
            selector=EpochSelector(CNN_NAMES)
            def log_cnn(stage):
                def log(epoch,losses):
                    for name,value in zip((*CNN_NAMES,'discriminator'),losses): training_log.writerow((run_seed,held_out,stage,epoch,name,value))
                return log
            def select_cnn(epoch,models):
                if epoch%args.check_every and epoch!=args.max_epochs: return False
                scores=validation_scores(dict(zip(CNN_NAMES,models)),val_examples,device,args.steps,70000+fold)
                done=selector.update(epoch,scores)
                for name,score in scores.items(): validation_log.writerow((run_seed,held_out,val_id,epoch,name,score,int(selector.stopped[name])))
                for stream in streams: stream.flush()
                print(f'Validation seed {run_seed}, test {held_out}, CNN epoch {epoch}',flush=True)
                return done
            torch.manual_seed(training_seed)
            inner_models=train(inner,args.max_epochs,device,np.random.default_rng(training_seed),log_cnn('inner'),select_cnn)
            del inner_models
            saits_selector=EpochSelector(('saits',))
            def select_saits(epoch,model):
                if epoch%args.check_every and epoch!=args.max_epochs:return False
                scores=validation_scores({'saits':model},val_examples,device,args.steps,70000+fold)
                done=saits_selector.update(epoch,scores)
                validation_log.writerow((run_seed,held_out,val_id,epoch,'saits',scores['saits'],int(done)))
                for stream in streams:stream.flush()
                return done
            saits_inner=train_saits(inner,args.max_epochs,device,training_seed+100000,
                                    select_saits,lambda e,l:training_log.writerow((run_seed,held_out,'inner',e,'saits',l)))
            del saits_inner
            selected={**selector.epochs,**saits_selector.epochs}
            for name in selected:
                sel=saits_selector if name=='saits' else selector
                selections.writerow((run_seed,held_out,val_id,name,selected[name],sel.best[name],int(sel.stopped[name])))
            snapshots={}
            def capture(epoch,models):
                for name,model in zip(CNN_NAMES,models):
                    if selected[name]==epoch:snapshots[name]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                return False
            torch.manual_seed(training_seed)
            refitted=train(outer,max(selected[n] for n in CNN_NAMES),device,np.random.default_rng(training_seed),log_cnn('refit'),capture)
            models=dict(zip(CNN_NAMES,refitted[:5]))
            for name,model in models.items():model.load_state_dict(snapshots[name]);model.eval()
            models['saits']=train_saits(outer,selected['saits'],device,training_seed+100000,
                                        lambda e,m:False,lambda e,l:training_log.writerow((run_seed,held_out,'refit',e,'saits',l)))
            torch.save({'models':{name:{k:v.detach().cpu() for k,v in model.state_dict().items()} for name,model in models.items()},
                        'selected_epochs':selected,'outer_training_subjects':outer_ids,'validation_subject':val_id},
                       checkpoint_dir/f'seed{run_seed}_subject{held_out}.pt')
            for (mask_set,seconds),(examples,classical) in test_cache.items():
                target,context,mask,scales=[v.numpy()[:,0] if v.ndim==3 else v.numpy() for v in examples.tensors]
                outputs={**classical,**predict(models,examples,device,args.steps,training_seed+100*mask_set)}
                rng=np.random.default_rng(training_seed+100*mask_set)
                outputs['gp']=np.stack([gp.sample(c,g,rng) for c,g in zip(context,mask)])
                for name,output in outputs.items():
                    if not np.isfinite(output).all():raise FloatingPointError(f'Invalid output {name}')
                    np.testing.assert_allclose(output[~mask],context[~mask],atol=1e-6)
                    errors=(np.abs(output-target)*mask).sum(axis=1)/mask.sum(axis=1)*scales
                    for tolerance in TOLERANCES:
                        for prominence in PROMINENCES:
                            stats=np.array([peak_stats(t,o,g,tolerance,prominence) for t,o,g in zip(target,output,mask)],dtype=float)
                            totals=stats.sum(axis=0)
                            sensitivities.writerow((run_seed,held_out,mask_set,seconds,name,tolerance,prominence,len(target),*totals))
                            if tolerance==0.15 and prominence==0.35:
                                metrics.writerow((run_seed,held_out,mask_set,seconds,name,len(target),float(errors.mean()),*totals))
                                for index,(error,values,gap) in enumerate(zip(errors,stats,mask)):
                                    windows.writerow((run_seed,held_out,mask_set,seconds,name,index,int(np.flatnonzero(gap)[0]),float(error),*values))
                np.savez_compressed(prediction_dir/f'seed{run_seed}_subject{held_out}_mask{mask_set}_gap{seconds}.npz',
                                    target=target,context=context,mask=mask,scales=scales,**{n:o.astype(np.float32) for n,o in outputs.items()})
            for stream in streams:stream.flush()
            manifest['completed_folds'].append([run_seed,held_out])
            manifest['folds'].append({'seed':run_seed,'test_subject':held_out,'inner_training_subjects':inner_ids,
                                      'validation_subject':val_id,'outer_training_subjects':outer_ids,'selected_epochs':selected})
            manifest['elapsed_seconds']=time.perf_counter()-started;save_manifest()
            print(f'FINISHED seed {run_seed}, test subject {held_out}; selected epochs {selected}',flush=True)
            del models,refitted
    manifest['status']='complete';manifest['elapsed_seconds']=time.perf_counter()-started;save_manifest()
    for stream in streams:stream.close()


if __name__=='__main__':main()
