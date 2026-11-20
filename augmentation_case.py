"""Execute one unchanged scientific augmentation protocol with checked faster execution."""
import hashlib
from pathlib import Path
import numpy as np
import torch
import synthetic_augmentation as experiment
import validated_benchmark as benchmark
from augmentation_engine import make_examples, train
from validated_benchmark import CNN_NAMES, EpochSelector, validation_scores, train_saits


def train_condition(condition, inner, outer, validation, seed, fold, args, paths, log):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_seed = 1000*seed+fold
    selector = EpochSelector(CNN_NAMES)

    def logger(stage):
        def write(epoch, values):
            if np.isscalar(values): values = {'saits':values}
            for method,value in values.items(): log('training',(condition,stage,epoch,method,value))
        return write

    def select(epoch, models):
        if epoch%args.check_every and epoch!=args.max_epochs: return False
        scores=validation_scores(dict(zip(CNN_NAMES,models)),validation,device,args.steps,70000+fold)
        done=selector.update(epoch,scores)
        for method,score in scores.items(): log('validation',(condition,epoch,method,score,int(selector.stopped[method])))
        print(f'{condition}: inner CNN epoch {epoch}; '+', '.join(f'{m}={scores[m]:.2f}' for m in CNN_NAMES),flush=True)
        return done

    torch.manual_seed(training_seed)
    models=train(inner,args.max_epochs,device,np.random.default_rng(training_seed),logger('inner'),select,
                 active=lambda method,epoch:not selector.stopped[method])
    del models
    saits_selector=EpochSelector(('saits',))
    def select_saits(epoch,model):
        if epoch%args.check_every and epoch!=args.max_epochs:return False
        scores=validation_scores({'saits':model},validation,device,args.steps,70000+fold)
        done=saits_selector.update(epoch,scores)
        log('validation',(condition,epoch,'saits',scores['saits'],int(done)))
        print(f'{condition}: inner SAITS epoch {epoch}; MAE={scores["saits"]:.2f}',flush=True)
        return done
    saits=train_saits(inner,args.max_epochs,device,training_seed+100000,select_saits,logger('inner'))
    del saits
    epochs={**selector.epochs,**saits_selector.epochs}; selection=[]
    for method in experiment.NEURAL:
        rule=saits_selector if method=='saits' else selector
        record={'condition':condition,'method':method,'selected_epoch':epochs[method],
                'validation_mae':rule.best[method],'stopped_by_patience':bool(rule.stopped[method])}
        selection.append(record);print('SELECTED',record,flush=True)
    snapshots={}
    def capture(epoch,models):
        for name,model in zip(CNN_NAMES,models):
            if epochs[name]==epoch:snapshots[name]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch%10==0:print(f'{condition}: refit CNN epoch {epoch}/{max(epochs[m] for m in CNN_NAMES)}',flush=True)
        return False
    torch.manual_seed(training_seed)
    fitted=train(outer,max(epochs[m] for m in CNN_NAMES),device,np.random.default_rng(training_seed),logger('refit'),capture,
                 active=lambda method,epoch:epoch<=epochs[method])
    models=dict(zip(CNN_NAMES,fitted))
    for name,model in models.items():model.load_state_dict(snapshots[name]);model.eval()
    models['saits']=train_saits(outer,epochs['saits'],device,training_seed+100000,lambda e,m:False,logger('refit'))
    torch.save({'models':{m:{k:v.detach().cpu() for k,v in model.state_dict().items()} for m,model in models.items()},
                'selection':selection},paths/f'{condition}.pt')
    print(f'FINISHED training {condition}',flush=True)
    return selection


if __name__=='__main__':
    experiment.make_examples=make_examples
    benchmark.make_examples=make_examples
    experiment.train_condition=train_condition
    original_save=experiment.save_json
    execution_sources=[Path(__file__),Path('augmentation_engine.py')]
    execution_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in execution_sources}
    def save_manifest(path,value):
        value['execution_code_sha256']=execution_hashes
        value['execution']='Batched bitwise-equivalent preprocessing; inactive independent CNN updates omitted with training RNG preserved. FP32 CUDA numerical equivalence checked; no bitwise GPU claim.'
        original_save(path,value)
    experiment.save_json=save_manifest
    experiment.main()
