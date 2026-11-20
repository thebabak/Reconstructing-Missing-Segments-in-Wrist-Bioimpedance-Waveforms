"""Participant-level augmentation effect figure, using the bundled runtime."""
from pathlib import Path
import csv,json,math
import numpy as np
from reportlab.graphics.shapes import Drawing,String,Rect,Line,Circle
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor,white,black

ROOT=Path(__file__).resolve().parents[1]
PREFIX=ROOT/'outputs/bioz_augmentation_multiseed'
OUT=ROOT/'work/augmentation_multiseed_figures';OUT.mkdir(parents=True,exist_ok=True)
assert json.loads(PREFIX.with_suffix('.audit.json').read_text())['status']=='passed'
with PREFIX.with_suffix('.paired.csv').open(newline='') as handle:paired=list(csv.DictReader(handle))
with PREFIX.with_suffix('.subjects.csv').open(newline='') as handle:subjects=list(csv.DictReader(handle))
names={'unet':'U-Net','saits':'SAITS compact','cvae':'Conditional VAE','cgan':'Conditional GAN','flow':'Flow matching','diffusion':'Diffusion'}
rows={r['method']:r for r in paired if r['gap_seconds']=='2'}
values=[];differences={}
for method in names:
    differences[method]=[]
    for subject in '123456':
        real,mixed=[next(r for r in subjects if r['condition']==c and r['method']==method and r['subject']==subject and r['gap_seconds']=='2') for c in ('real_only_matched','real_plus_synthetic')]
        differences[method].append(float(mixed['mae'])-float(real['mae']))
    values.extend(differences[method]);values.extend(float(rows[method][k]) for k in ('mae_bootstrap_low','mae_bootstrap_high'))
step=max(1,math.ceil((max(values)-min(values))/8))
low=math.floor(min(min(values),0)/step)*step-step/2
high=math.ceil(max(max(values),0)/step)*step+step/2
d=Drawing(820,470);d.add(Rect(0,0,820,470,fillColor=white,strokeColor=None))
d.add(String(18,442,'Paired effect of synthetic augmentation at 2 s',fontName='Helvetica-Bold',fontSize=20))
d.add(String(18,420,'Synthetic minus real-only MAE; negative values favor augmentation',fontName='Helvetica',fontSize=15))
x0,width=165,390
def x(value):return x0+(value-low)/(high-low)*width
for tick in np.arange(math.ceil(low/step)*step,high+1e-9,step):
    xx=x(tick);d.add(Line(xx,91,xx,385,strokeColor=HexColor('#dddddd'),strokeWidth=.6))
    d.add(String(xx,72,f'{tick:g}',textAnchor='middle',fontName='Helvetica',fontSize=13))
d.add(Line(x(0),91,x(0),385,strokeColor=black,strokeWidth=1))
d.add(String(580,391,'Mean [95% interval]',fontName='Helvetica-Bold',fontSize=13))
for index,method in enumerate(names):
    y=360-index*47;r=rows[method]
    estimate=float(r['mae_difference']);a=float(r['mae_bootstrap_low']);b=float(r['mae_bootstrap_high'])
    color=HexColor('#176b55' if estimate<0 else '#aa5d25')
    d.add(String(x0-12,y-3,names[method],textAnchor='end',fontName='Helvetica',fontSize=15))
    for j,value in enumerate(differences[method]):
        d.add(Circle(x(value),y+(j-2.5)*2.3,2.5,fillColor=HexColor('#b2b2b2'),strokeColor=None))
    d.add(Line(x(a),y,x(b),y,strokeColor=color,strokeWidth=2))
    for endpoint in (a,b):d.add(Line(x(endpoint),y-5,x(endpoint),y+5,strokeColor=color,strokeWidth=1.3))
    d.add(Circle(x(estimate),y,4.3,fillColor=color,strokeColor=white,strokeWidth=.6))
    d.add(String(580,y-3,f'{estimate:+.2f} [{a:+.2f}, {b:+.2f}]',fontName='Helvetica',fontSize=15))
d.add(String(x0+width/2,50,'MAE difference (mOhm)',textAnchor='middle',fontName='Helvetica',fontSize=15))
d.add(String(18,25,'Gray points: six participant differences after averaging five runs and three masks.',fontName='Helvetica',fontSize=13))
d.add(String(18,10,'Intervals: exploratory percentile participant bootstrap (10,000 resamples), conditional on fitted models.',fontName='Helvetica',fontSize=13))
renderSVG.drawToFile(d,str(OUT/'effects.svg'))
print('Created participant effect figure')
