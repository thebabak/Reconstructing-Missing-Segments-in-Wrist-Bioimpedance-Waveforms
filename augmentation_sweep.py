"""Run every held-out participant and five seeds, preserving per-case artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from synthetic_augmentation import save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/bioz_augmentation_multiseed'))
    parser.add_argument('--data',type=Path,default=Path('work/physionet_bioz'))
    parser.add_argument('--subjects',nargs='+',default=list('123456'))
    parser.add_argument('--seeds',nargs='+',type=int,default=list(range(2026,2031)))
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--max-epochs',type=int,default=120)
    args=parser.parse_args()
    if not 1<=args.workers<=2:parser.error('One or two workers supported')
    root=Path(__file__).resolve().parent
    case_dir=Path('outputs')/(args.output.name+'_cases');case_dir.mkdir(parents=True,exist_ok=True)
    log_dir=Path('work')/(args.output.name+'_logs');log_dir.mkdir(parents=True,exist_ok=True)
    manifest_path=args.output.with_suffix('.run.json')
    sources=[Path(__file__),Path('augmentation_case.py'),Path('augmentation_engine.py'),Path('synthetic_augmentation.py'),Path('bioz_inpainting.py'),Path('validated_benchmark.py')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    cases=[]
    old=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if old:assert old['code_sha256']==hashes,'Source changed; use a new output prefix'
    for subject in args.subjects:
        for seed in args.seeds:
            prior=next((c for c in old['cases'] if c['subject']==subject and c['seed']==seed),None) if old else None
            attempt=prior['attempt'] if prior else 1
            prefix=Path(prior['prefix']) if prior else case_dir/f'subject{subject}_seed{seed}'
            if prefix.with_suffix('.run.json').exists():
                existing=json.loads(prefix.with_suffix('.run.json').read_text())
                if existing['status']=='complete':
                    cases.append({'subject':subject,'seed':seed,'attempt':attempt,'prefix':str(prefix),'status':'complete'})
                    continue
                attempt+=1;prefix=case_dir/f'subject{subject}_seed{seed}_attempt{attempt}'
            cases.append({'subject':subject,'seed':seed,'attempt':attempt,'prefix':str(prefix),'status':'pending'})
    state={'status':'running','subjects':args.subjects,'seeds':args.seeds,'workers':args.workers,'max_epochs':args.max_epochs,
           'conditions':['real_only_matched','real_plus_synthetic'],'methods':['unet','flow','diffusion','cvae','cgan','saits'],
           'expected_cases':len(cases),'cases':cases,'code_sha256':hashes,'started_unix':time.time(),
           'protocol':'Each held-out participant, five rotating training-only validation identities via seeds2026-2030; fresh 1:1 synthetic augmentation versus epoch-size-matched real control; cap120, checks5, patience3; refit; three test masks and three gaps.',
           'pilot':'Earlier single-person pilot informed this expanded experiment; all cases are rerun with the same prespecified simulator and objectives, using numerically checked faster execution. The same six-person cohort is reused.'}
    running={};last_report=0
    while True:
        for case in cases:
            if len(running)>=args.workers:break
            if case['status']!='pending':continue
            logfile=log_dir/(Path(case['prefix']).name+'.log')
            stream=logfile.open('w',encoding='utf-8')
            command=[sys.executable,'-u',str(root/'augmentation_case.py'),'--data',str(args.data),'--output',case['prefix'],
                     '--test-subject',case['subject'],'--seed',str(case['seed']),'--max-epochs',str(args.max_epochs)]
            env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1'
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            case.update(status='running',pid=process.pid,log=str(logfile))
            running[process.pid]=(process,stream,case)
            print(f'START subject {case["subject"]}, seed {case["seed"]}, PID {process.pid}',flush=True)
        for pid,(process,stream,case) in list(running.items()):
            result=process.poll()
            if result is None:continue
            stream.close();del running[pid]
            case['status']='complete' if result==0 else 'failed';case['exit_code']=result
            if result==0:
                completed=json.loads(Path(case['prefix']).with_suffix('.run.json').read_text())
                assert completed['status']=='complete'
                case['elapsed_seconds']=completed['elapsed_seconds']
            print(f'{case["status"].upper()} subject {case["subject"]}, seed {case["seed"]}',flush=True)
        state['completed_cases']=sum(c['status']=='complete' for c in cases)
        state['elapsed_seconds']=time.time()-state['started_unix']
        if time.time()-last_report>=30:
            active=[]
            for _,_,case in running.values():
                try:
                    tail=Path(case['log']).read_text(encoding='utf-8').splitlines()[-1]
                except (OSError,IndexError):tail='initializing'
                active.append(f'subject{case["subject"]}/seed{case["seed"]}: {tail[:160]}')
            print(f'PROGRESS {state["completed_cases"]}/{len(cases)}; '+' | '.join(active),flush=True)
            last_report=time.time()
        save_json(manifest_path,state)
        if not running and not any(c['status']=='pending' for c in cases):break
        time.sleep(2)
    state['status']='complete' if all(c['status']=='complete' for c in cases) else 'failed'
    save_json(manifest_path,state)
    print('SWEEP',state['status'],state['completed_cases'],'cases',state['elapsed_seconds'],'seconds',flush=True)
    if state['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
