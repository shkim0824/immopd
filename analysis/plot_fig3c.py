"""Figure 3c: average normalized score during MOPD for different merge initializations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
D = json.loads((HERE / "data" / "fig3c.json").read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':12,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
specs=[('nonuniform_2.1','#D65470','-','o',r'Non-uniform, $\lambda=2.1$'),('nonuniform_1','#D65470','--','o',r'Non-uniform, $\lambda=1$'),('uniform_2.1','#3978A8','-','s',r'Uniform, $\lambda=2.1$'),('uniform_1','#3978A8','--','s',r'Uniform, $\lambda=1$'),('base','#777777','-','^','Base')]
fig=plt.figure(figsize=(.95*2.94*.40/.29,.95*2.65));ax=fig.add_axes([.17,.48/2.65,.80,(2.65-.48-.52)/2.65])
for key,col,ls,mark,label in specs:
 points=D[key];xs=[int(x) for x in points];ys=list(points.values())
 ax.plot(xs,ys,color=col,ls=ls,marker=mark,lw=1.5,ms=4,label=label,markerfacecolor='white' if ls=='--' else col,zorder=4 if key=='uniform_2.1' else 3)
ax.set(xlim=(-3,105),ylim=(-3,87),xticks=[0,25,50,75,100],yticks=[0,20,40,60,80],xlabel='MOPD step',ylabel='Norm. score (%)')
ax.grid(color='#e9e9e9',lw=.6);ax.set_axisbelow(True)
handles,labels=ax.get_legend_handles_labels();order=[0,1,2,3,4]
rows=[[0,2,4],[1,3]]
legends=[]
y=2.61/2.65
for row in rows:
 legend=fig.legend(handles=[handles[i] for i in row],loc='upper center',bbox_to_anchor=(.5,y),ncol=len(row),frameon=False,fontsize=9,handlelength=1.55,handletextpad=.4,columnspacing=.8,labelspacing=.15,borderaxespad=0,borderpad=.2)
 legends.append(legend)
 fig.canvas.draw();ren=fig.canvas.get_renderer()
 y=(legend.get_window_extent(ren).y0-1.35/72*fig.dpi)/fig.bbox.height
legend=legends[-1]
gap=3/72*fig.dpi;excess=ax.get_window_extent(ren).y1+gap-legend.get_window_extent(ren).y0
if excess>0:
 pos=ax.get_position();ax.set_position([pos.x0,pos.y0,pos.width,pos.height-excess/fig.bbox.height])
fig.canvas.draw();ren=fig.canvas.get_renderer()
for legend in legends:
 bounds=legend.get_window_extent(ren);canvas=fig.get_window_extent(ren)
 assert abs((bounds.x0+bounds.x1)/2-canvas.width/2)<.01
 assert bounds.x0>=0 and bounds.x1<=canvas.x1 and bounds.y0>ax.get_window_extent(ren).y1
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig3c.{ext}", dpi=300)
plt.close(fig)
