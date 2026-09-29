"""Figure 4: per-domain normalized score during MOPD."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
D = json.loads((HERE / "data" / "fig4.json").read_text())
plt.rcParams.update({'pdf.fonttype':42})
RUNS=D['runs'];ours_ck={int(k):v for k,v in D['immopd_checkpoint'].items()};ours_mg={int(k):v for k,v in D['immopd_postmerge'].items()}
PANELS=[('medqa','Medical'),('tau2','Tool-Use'),('finqa','Finance'),('casehold','Law'),('ifeval','IF')]
XMAX=100;YLO=-.4;SEL='';MERGES=(25,50,75)
COL={'base':'#777777','sft':'#4477AA','merge':'#CCBB44','ours':'#EE6677'}
BIG = len(PANELS) == 5
FS = dict(title=10.5, tick=9, label=10.5, legend=9, note=8.5, ms=6) if BIG else dict(title=10, tick=9, label=10, legend=8.5, note=7.5, ms=5)
FIGSIZE = (8.5, 1.8) if BIG else (2.55 * len(PANELS) + 0.0, 2.75)
fig, axes = plt.subplots(1, len(PANELS), figsize=FIGSIZE, gridspec_kw={"wspace": 0.22 if BIG else 0.28}, squeeze=False)
axes = axes[0]
for ax, (k, title) in zip(axes, PANELS):
    ax.axhline(0, color="#c3c2b7", lw=1, zorder=1)
    ax.axhline(1, color="#9a9890", lw=1, ls="--", zorder=1)
    for name in ("base", "sft", "merge"):
        pts = [(s, p[k]) for s, p in RUNS[name] if k in p and s <= XMAX]
        if pts:
            xs, ys = zip(*pts)
            yc = [max(y, YLO + 0.03) for y in ys]
            ax.plot(xs, yc, color=COL[name], lw=1.6, zorder=2)
            for x, y, y2 in zip(xs, ys, yc):
                if y < YLO + 0.03:
                    ax.scatter([x], [y2], marker="v", s=30, color=COL[name], zorder=3, clip_on=False)
                    ax.annotate(f"{y:.2f}", (x, y2), textcoords="offset points", xytext=(12, 5) if BIG else (5, 0), va="center",
                                fontsize=FS["note"], color=COL[name])
                else:
                    ax.scatter([x], [y], s=12, color=COL[name], zorder=3)
    c = COL["ours"]
    seq = []
    for s in sorted(x for x in ours_ck if x <= XMAX):
        if k in ours_ck[s]:
            seq.append((s, ours_ck[s][k], "pre" if s in MERGES else "ck"))
        if s in MERGES and k in ours_mg.get(s, {}):
            seq.append((s, ours_mg[s][k], "post"))
    for (x0, y0, k0), (x1, y1, k1) in zip(seq, seq[1:]):
        if x0 == x1 and k0 == "pre" and k1 == "post":
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=5,
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.3, mutation_scale=8, shrinkA=3, shrinkB=3))
        else:
            ax.plot([x0, x1], [y0, y1], color=c, lw=2.0, zorder=4)
    for x, y, kd in seq:
        ax.scatter([x], [y], s=26 if kd != "ck" else 14, facecolor="white" if kd == "pre" else c,
                   edgecolor=c, lw=1.4, zorder=6)
    ax.set_title(title, fontsize=FS["title"], pad=4)
    ax.set_xlim(-0.04 * XMAX, 1.04 * XMAX); ax.set_xticks(list(range(0, XMAX + 1, (50 if BIG else 25) if XMAX <= 100 else 50)))
    ax.set_ylim(YLO, 1.2); ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=FS["tick"], length=3, pad=3 if BIG else 2)
    if BIG and ax is not axes[0]:
        ax.tick_params(labelleft=False)
    ax.grid(axis="y", color="#eeede9", lw=0.8, zorder=0); ax.set_axisbelow(True)
    if BIG:
        ax.tick_params(length=0)
    else:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Norm. score" if BIG else "Normalized score", fontsize=FS["label"])
fig.supxlabel("MOPD step", fontsize=FS["label"], y=-0.1 if BIG else -0.04)
fig.legend(handles=[Line2D([], [], color=COL["base"], lw=1.6, label="Base-init"),
                    Line2D([], [], color=COL["sft"], lw=1.6, label="SFT warm-up"),
                    Line2D([], [], color=COL["merge"], lw=1.6, label="Uniform Merge"),
                    Line2D([], [], color=COL["ours"], lw=2.0, label="IM-MOPD"),
                    Line2D([], [], color=COL["ours"], marker="o", mfc="white", ls="none", markersize=FS["ms"], label="Before merge"),
                    Line2D([], [], color=COL["ours"], marker="o", ls="none", markersize=FS["ms"], label="After merge")],
           loc="upper center", bbox_to_anchor=(0.5, -0.11 if BIG else {2: 1.2, 3: 1.13}[len(PANELS)]), ncol={2: 2, 3: 3, 5: 6}[len(PANELS)], fontsize=FS["legend"], frameon=False,
           columnspacing=0.8 if BIG else 1.0, handlelength=1.4 if BIG else 1.6, handletextpad=0.4)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig4.{ext}", dpi=300, bbox_inches="tight")
