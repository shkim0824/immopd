"""Figure 1: normalized score before and after MOPD for each initialization."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
D = json.loads((HERE / "data" / "fig1.json").read_text())
plt.rcParams.update({'pdf.fonttype':42})
PANELS=[('avg','5-domain avg.'),('tau2','Tool-Use'),('medqa','Medical'),('finqa','Finance')]
AVG=['tau2','medqa','finqa','casehold','ifeval']
FLOOR={'tau2':5.263157894736842,'medqa':67.9,'finqa':60.8,'casehold':62.12,'ifeval':37.52}
REF={'tau2':66.66666666666667,'medqa':79.3,'finqa':75.3,'casehold':73.64,'ifeval':77.63}
COL={'base':'#777777','sft':'#4477AA','merge':'#CCBB44','immopd':'#EE6677'}
GROUP={'base':0,'sft':1,'merge':2,'immopd':3}
FS=dict(title=10.5,tick=9,label=10.5,legend=10.5,rank=7.5);MS=135
data=[(r['label'],r['family'],r['init'],r['final']) for r in D]
def r(q, d):
    if d == "avg":
        return sum(r(q, k) for k in AVG) / len(AVG)
    return (q[d] - FLOOR[d]) / (REF[d] - FLOOR[d])
n, ys = len(data), list(range(len(data)))[::-1]

def make_ranks(values):
    ranks, prev = {}, (None, None)
    for pos, (lab, s) in enumerate(sorted(values.items(), key=lambda x: x[1], reverse=True), start=1):
        ranks[lab] = prev[1] if s == prev[0] else pos
        prev = (s, ranks[lab])
    return ranks
rank_i = {d: make_ranks({lab: r(q0, d) for lab, _, q0, _ in data}) for d, _ in PANELS}
rank_f = {d: make_ranks({lab: r(q1, d) for lab, _, _, q1 in data}) for d, _ in PANELS}

def panel(ax, vals, title, xlim, rs, re):
    min_sep = 0.115 * (xlim[1] - xlim[0])
    ax.axvline(0, color="#c3c2b7", lw=1, zorder=1)
    ax.axvline(1, color="#9a9890", lw=1, ls="--", zorder=1)
    for y, (a, b), (lab, fam, _, _) in zip(ys, vals, data):
        c, crowded = COL[fam], abs(b - a) < min_sep
        ax.annotate("", xy=(b, y), xytext=(a, y), zorder=2,
                    arrowprops=dict(arrowstyle="-", color=c, lw=1.8, shrinkA=7, shrinkB=7))
        ax.scatter([a], [y], s=MS, facecolor="white", edgecolor=c, linewidth=1.8, zorder=4)
        ax.scatter([b], [y], s=MS, facecolor=c, edgecolor=c, linewidth=1.8, zorder=4)
        if crowded:
            ax.text(max(a, b) + 0.045 * (xlim[1] - xlim[0]), y, f"{rs[lab]}→{re[lab]}", fontsize=FS["rank"],
                    fontweight="bold", color=c, ha="left", va="center", zorder=5)
        else:
            ax.text(a, y, str(rs[lab]), fontsize=FS["rank"], fontweight="bold", color=c, ha="center", va="center", zorder=5)
            ax.text(b, y, str(re[lab]), fontsize=FS["rank"], fontweight="bold", color="white", ha="center", va="center", zorder=5)
    for k in range(1, n):
        if GROUP[data[k][1]] != GROUP[data[k - 1][1]]:
            ax.axhline(ys[k] + 0.5, color="#dedcd4", lw=0.8, ls=":", zorder=0)
    ax.set_xlim(*xlim); ax.set_ylim(-0.65, n - 0.35)
    ax.set_title(title, fontsize=FS["title"], pad=4)
    ax.grid(axis="x", color="#eeede9", lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks(ys); ax.set_yticklabels([])
    ax.tick_params(length=0, labelsize=FS["tick"], pad=3)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

WS = 0.08
FIGW = 10.5 * (len(PANELS) + WS * (len(PANELS) - 1)) / (4 + WS * 3)
fig, axes = plt.subplots(1, len(PANELS), figsize=(FIGW, 2.5), gridspec_kw={"wspace": WS})
for ax, (d, title) in zip(axes, PANELS):
    vals = [(r(q0, d), r(q1, d)) for _, _, q0, q1 in data]
    lo = min(-0.25, min(min(v) for v in vals) - 0.12)
    panel(ax, vals, title, (lo, 1.18), rank_i[d], rank_f[d])
axes[0].set_yticklabels([l for l, *_ in data], fontsize=FS["tick"])
bb = fig.get_tightbbox(fig.canvas.get_renderer())
XC = (bb.x0 + bb.x1) / 2 / fig.get_figwidth()
fig.supxlabel("Norm. score", fontsize=FS["label"], x=XC, y=-0.05)
fig.legend(handles=[Line2D([], [], color=COL[k], marker="o", mfc="white", mew=1.8, label=v)
                    for k, v in (("base", "No init"), ("sft", "SFT warm-up init"),
                                 ("merge", "Task-vector merge init"), ("immopd", "IM-MOPD"))],
           loc="upper center", bbox_to_anchor=(XC, -0.05), fontsize=FS["legend"], frameon=False,
           ncol=4, columnspacing=1.2, handletextpad=0.4)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig1.{ext}", dpi=300, bbox_inches="tight")
