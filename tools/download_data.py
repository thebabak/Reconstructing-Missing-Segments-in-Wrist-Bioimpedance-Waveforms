"""Download only the public CSV files named by the recorded dataset manifest."""
from pathlib import Path
from urllib.request import Request, urlopen
import argparse,hashlib,json

ROOT=Path(__file__).resolve().parents[1]
BASE='https://physionet.org/files/bp-graphene-bioimpedance/1.0.0/'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list',action='store_true',help='List expected files without downloading')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'results/reference/outputs/bioz_metrics_validated.run.json').read_text())
    for path,digest in sorted(manifest['data_sha256'].items()):
        relative=Path(path)
        assert relative.parts[:2]==('work','physionet_bioz') and '..' not in relative.parts
        target=(ROOT/relative).resolve();assert target.is_relative_to(ROOT)
        if args.list:
            print(relative.as_posix(),digest);continue
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==digest:
            print('Verified cached:',relative.as_posix());continue
        if target.exists():raise RuntimeError('Existing file has a different SHA-256: '+str(relative))
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix(target.suffix+'.part')
        request=Request(BASE+Path(*relative.parts[2:]).as_posix(),headers={'User-Agent':'WristBioZ-Reproducibility/0.1'})
        print('Downloading:',relative.as_posix(),flush=True)
        checksum=hashlib.sha256()
        try:
            with urlopen(request,timeout=90) as response,temporary.open('wb') as output:
                while chunk:=response.read(1024*1024):
                    checksum.update(chunk);output.write(chunk)
            if checksum.hexdigest()!=digest:raise RuntimeError('Downloaded SHA-256 differs: '+str(relative))
            temporary.replace(target)
        finally:
            if temporary.exists():temporary.unlink()
    print('Listed' if args.list else 'Verified',len(manifest['data_sha256']),'dataset files.')

if __name__=='__main__':main()
