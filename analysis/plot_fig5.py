"""Figure 5: teacher continuation from student prefixes (a: MedQA, b: tau2-Telecom)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
D = json.loads((HERE / "data" / "fig5.json").read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':12,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
lo=min(p['mean']-(p['sd'] or 0) for ps in D['medical'].values() for p in ps)-1.;tp=D['medical']['merge_init'][0];hi=tp['mean']+tp['sd']+1.2
specs=[('medical','fig5a',[('merge_init','#E96883','Merge initialization'),('sft_warmup','#4477AA','SFT warm-up initialization')]),('tooluse','fig5b',[('immopd','#E96883','Iterative merging + MOPD'),('sft_warmup','#4477AA','SFT warm-up + MOPD')])]
def draw(ax,domain,curves):
 for key,col,label in curves:
  ps=D[domain][key];x=np.array([p['x'] for p in ps]);y=np.array([p['mean'] for p in ps]);sd=np.array([p['sd'] or 0 for p in ps])
  ax.plot(x,y,color=col,lw=2,label=label);ax.scatter(x[1:-1],y[1:-1],color=col,s=20,zorder=4);ax.plot(100,y[-1],marker='o',ms=4.5,mfc='white',mec=col,zorder=5)
 teacher=D[domain][curves[0][0]][0]['mean'];ax.axhline(teacher,color='#777777',ls=':',lw=1,zorder=1);ax.plot(0,teacher,marker='*',color='#777777',ms=8,zorder=5)
 ax.annotate('Teacher only',(0,teacher),xytext=(5,8),textcoords='offset points',fontsize=9,color='#555555')
 ax.set(xlim=(-4,104),ylim=(lo,hi) if domain=='medical' else (20,80),xticks=[0,10,30,50,70,100],yticks=np.arange(np.ceil(lo/2)*2,hi,2) if domain=='medical' else np.arange(20,81,10),xlabel='Student prefix (%)',ylabel='Acc. (%)' if domain=='medical' else 'Task success (%)')
 ax.tick_params(length=3);ax.grid(axis='y',color='#e9e9e9',lw=.65);ax.set_axisbelow(True);ax.spines['left'].set_color('#888888');ax.spines['bottom'].set_color('#888888')
 leg=ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),frameon=False,handlelength=2,labelspacing=.4,borderaxespad=0)
 return leg
for domain,name,curves in specs:
 fig=plt.figure(figsize=(3.65,3.2));ax=fig.add_axes([.19,.20,.785,.59]);draw(ax,domain,curves)
 for ext in ('pdf','png'):fig.savefig(OUT/f'{name}.{ext}',dpi=300)
 plt.close(fig)
