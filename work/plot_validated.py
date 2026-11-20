"""Scientific plots using ReportLab chart primitives from the bundled runtime."""
import csv,json
from pathlib import Path
import numpy as np
from reportlab.graphics.shapes import Drawing,String,Rect,Circle
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor,Color,white,black

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/validated_figures';OUT.mkdir(exist_ok=True)
NAMES={'unet':'U-Net','saits':'SAITS compact','cvae':'Conditional VAE','cgan':'Conditional GAN',
       'flow':'Flow matching','diffusion':'Diffusion','gp':'GP sample','gp_mean':'GP mean',
       'periodic':'Periodic harmonic','pchip':'PCHIP','prefill':'Filtered prefill','linear':'Linear'}
COLORS=['#1764a0','#d17b13','#288348','#aa385c','#7356a3','#347d84']
METHOD_COLORS=dict(zip(('unet','saits','cvae','cgan','flow','diffusion'),COLORS))
METHOD_COLORS.update(gp_mean='#aa4499',periodic='#117733')

def panel(d,x,y,w,h,data,labels,title,xlabel,ylabel,xmin,xmax,ymin=None,ymax=None,palette=None):
 palette=palette or COLORS
 lp=LinePlot();lp.x=x;lp.y=y;lp.width=w;lp.height=h;lp.data=data;lp.joinedLines=1
 lp.xValueAxis.valueMin=xmin;lp.xValueAxis.valueMax=xmax
 lp.xValueAxis.labels.fontSize=8.5;lp.yValueAxis.labels.fontSize=8.5
 lp.xValueAxis.strokeColor=HexColor('#888888');lp.yValueAxis.strokeColor=HexColor('#888888')
 if ymin is not None:lp.yValueAxis.valueMin=ymin
 if ymax is not None:lp.yValueAxis.valueMax=ymax
 for i in range(len(data)):
  lp.lines[i].strokeColor=HexColor(palette[i%len(palette)]);lp.lines[i].strokeWidth=1.15
 d.add(lp);d.add(String(x,y+h+13,title,fontName='Helvetica-Bold',fontSize=10))
 d.add(String(x+w/2,y-27,xlabel,textAnchor='middle',fontSize=9))
 d.add(String(5,y+h+2,ylabel,fontSize=8.5))
 for i,label in enumerate(labels):
  col=i%3;row=i//3;xx=x+col*w/3;yy=y-43-row*13
  d.add(Rect(xx,yy-1,9,3,fillColor=HexColor(palette[i%len(palette)]),strokeColor=None))
  d.add(String(xx+13,yy-2,label,fontSize=8.5))
 return lp


def make_figures():
 rows=list(csv.DictReader((ROOT/'outputs/bioz_metrics_validated.validation.csv').open()))
 selections=list(csv.DictReader((ROOT/'outputs/bioz_metrics_validated.selection.csv').open()))
 methods=('unet','saits','cvae','cgan','flow','diffusion')
 curves=[];selected=[]
 for m in methods:
  a=[r for r in rows if r['seed']=='2026' and r['held_out_subject']=='1' and r['method']==m]
  active=[]
  for r in a:
   active.append((float(r['epoch']),float(r['validation_mae'])))
   if r['stopped']=='1':break
  curves.append(active)
  s=next(r for r in selections if r['seed']=='2026' and r['held_out_subject']=='1' and r['method']==m)
  selected.append((float(s['selected_epoch']),float(s['validation_mae'])))
 d=Drawing(480,305)
 lp=panel(d,49,77,415,175,curves,[NAMES[m] for m in methods],
          'Training-only validation example','Epoch','Validation MAE (mOhm)',0,60,0,None)
 ymin=0.; ymax=max(y for curve in curves for _,y in curve)*1.1
 lp.yValueAxis.valueMax=ymax
 for i,(epoch,value) in enumerate(selected):
  d.add(Circle(49+epoch/60*415,77+value/ymax*175,3,fillColor=HexColor(COLORS[i]),strokeColor=white,strokeWidth=.5))
 d.add(String(49,5,'Markers identify the selected epoch. Test subject 1, seed 2026.',fontSize=8.5))
 renderSVG.drawToFile(d,str(OUT/'validation.svg'))

 p=ROOT/'work/bioz_metrics_validated_predictions/seed2026_subject1_mask0_gap2.npz'
 a=np.load(p);mask=a['mask'];target=a['target'];scale=a['scales']
 errors=(np.abs(a['unet']-target)*mask).sum(axis=1)/mask.sum(axis=1)*scale
 index=int(np.argsort(errors)[len(errors)//2]);gap=np.flatnonzero(mask[index]);start=gap[0]/125;end=(gap[-1]+1)/125
 lower=max(0,start-.75);upper=min(8,end+.75)
 t=np.arange(1000)/125;keep=(t>=lower)&(t<=upper)
 ref=target[index]*scale[index]
 d=Drawing(480,555)
 for j,methods2 in enumerate((('unet','gp_mean','periodic'),('saits','cgan','diffusion'))):
  y=335 if j==0 else 80
  arrays=[ref]+[a[m][index]*scale[index] for m in methods2]
  values=np.concatenate([v[keep] for v in arrays]);lo=float(values.min());hi=float(values.max());margin=(hi-lo)*.08
  d.add(Rect(49+(start-lower)/(upper-lower)*415,y,(end-start)/(upper-lower)*415,155,
             fillColor=HexColor('#eeeeee'),strokeColor=None))
  chart=panel(d,49,y,415,155,[list(zip(t[keep],v[keep])) for v in arrays],
              ['Intact reference']+[NAMES[m] for m in methods2],
              ('A  U-Net and classical baselines' if j==0 else 'B  Neural reconstructions'),
              'Time in window (s)','Centered amplitude (mOhm)',lower,upper,lo-margin,hi+margin,palette=['#000000',*[METHOD_COLORS[m] for m in methods2]])
  chart.lines[0].strokeColor=black;chart.lines[0].strokeWidth=1.7
 d.add(String(49,5,f'Subject 1, seed 2026, mask set 0, window {index}; gray band is the missing segment.',fontSize=8.5))
 renderSVG.drawToFile(d,str(OUT/'waveforms.svg'))
 (OUT/'example.json').write_text(json.dumps({'subject':'1','seed':2026,'mask_set':0,'gap_seconds':2,
       'window_index':index,'gap_start_seconds':start,'unet_mae':float(errors[index]),
       'selection':'Upper median U-Net gap MAE among subject 1 windows; first ordered seed and mask set.'},indent=2))

 sensitivity_path=ROOT/'outputs/bioz_metrics_validated.sensitivity_summary.csv'
 if not sensitivity_path.exists():
  print('Created preliminary validation and waveform figures; sensitivity awaits complete results.');return
 rows=list(csv.DictReader(sensitivity_path.open()))
 methods2=('linear','prefill','pchip','periodic','gp_mean','unet','saits','cvae','cgan','flow','diffusion','gp')
 settings=[(tol,prom) for tol in ('0.05','0.1','0.15') for prom in ('0.2','0.35','0.5')]
 d=Drawing(480,345)
 d.add(String(15,325,'Peak F1 sensitivity at a 2 s gap',fontName='Helvetica-Bold',fontSize=11))
 d.add(String(15,309,'Mean over five runs and three mask schedules; actual tolerance (ms) / prominence',fontSize=8.5))
 for col,(tol,prom) in enumerate(settings):
  d.add(String(141+col*36,289,f'{round(float(tol)*125)*8}/{prom}',fontSize=8.5,textAnchor='middle'))
 for row,m in enumerate(methods2):
  y=265-row*20
  d.add(String(118,y+5,NAMES[m],fontSize=8.5,textAnchor='end'))
  for col,(tol,prom) in enumerate(settings):
   vals=[float(r['f1']) for r in rows if r['method']==m and r['gap_seconds']=='2' and r['tolerance_seconds']==tol and r['prominence']==prom]
   value=float(np.mean(vals));x=123+col*36
   fill=Color(1-value*.88,1-value*.60,1-value*.28)
   d.add(Rect(x,y,35,19,fillColor=fill,strokeColor=white,strokeWidth=.5))
   d.add(String(x+17.5,y+6,f'{value:.2f}',fontSize=8.5,textAnchor='middle',fillColor=black))
 renderSVG.drawToFile(d,str(OUT/'sensitivity.svg'))
 print('Created three SVG figures and deterministic waveform selection metadata.')

if __name__=='__main__':make_figures()
