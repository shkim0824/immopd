"""Figure 3b: two-domain merge initializations before and after MOPD, and IM-MOPD."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
D = json.loads((HERE / "data" / "fig3b.json").read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':12,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(.95*2.94,.95*2.65));ax=fig.add_axes([.21,.48/2.65,.76,(2.65-.48-.36)/2.65])
for g,col in zip(D['grid'],['#70A6BE','#4985AA','#306389','#173E60']):
 a=g['init'];b=g['final'];x=[a['medqa'],b['medqa']];y=[a['ifeval'],b['ifeval']]
 ax.annotate('',xy=(x[1],y[1]),xytext=(x[0],y[0]),arrowprops={'arrowstyle':'->','color':col,'lw':1.1,'alpha':.7},zorder=2)
 ax.scatter(x[0],y[0],s=44,facecolor='white',edgecolor=col,lw=1.1,zorder=3);ax.scatter(x[1],y[1],s=44,color=col,zorder=4)
 ax.annotate(f"{g['r']:.1f}",(x[0],y[0]),textcoords='offset points',xytext=(0,6),ha='center',fontsize=9,color=col)
q=D['immopd']['final'];ax.scatter(q['medqa'],q['ifeval'],marker='*',s=240,color='#D84E6B',edgecolor='white',linewidth=.4,zorder=6)
ax.set(xlim=(72.5,81.2),ylim=(44,85.5),xticks=[74,76,78,80],yticks=[50,60,70,80],xlabel='MedQA acc. (%)',ylabel='IFEval acc. (%)');ax.grid(color='#ededed',lw=.65);ax.set_axisbelow(True)
handles=[Line2D([],[],marker='o',mfc='white',mec='#306389',color='none',label='Initialization'),Line2D([],[],marker='o',mfc='#306389',mec='#306389',color='none',label='After MOPD'),Line2D([],[],marker='*',markersize=13,mfc='#D84E6B',mec='#D84E6B',color='none',label='IM-MOPD')]
legend_x = .53 + .8 / (.29 * 5.5 * 25.4)
rows=[[0,1],[2]]
legends=[]
y=2.61/2.65
for row in rows:
 legend=fig.legend(handles=[handles[i] for i in row],loc='upper center',bbox_to_anchor=(legend_x,y),ncol=len(row),frameon=False,fontsize=9,handlelength=1.1,handletextpad=.4,columnspacing=.8,labelspacing=.15,borderaxespad=0,borderpad=.2)
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
 assert abs((bounds.x0+bounds.x1)/2-canvas.width*legend_x)<.01
 assert bounds.x0>=0 and bounds.x1<=canvas.x1 and bounds.y0>ax.get_window_extent(ren).y1
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig3b.{ext}", dpi=300)
plt.close(fig)
