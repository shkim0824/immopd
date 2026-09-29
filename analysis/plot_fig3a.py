"""Figure 3a: effect of a merge intervention and of the following 50 MOPD updates."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
D = json.loads((HERE / "data" / "fig3a.json").read_text())
plt.rcParams.update({'pdf.fonttype':42})
LABEL = {"mean": "Mean", "tau": "Tool", "med": "Med", "if": "IF", "law": "Law", "fin": "Fin"}
ORDER = ["mean", "tau", "med", "if", "law", "fin"]
R = {r["dom"]: r for r in D}
n = len(ORDER)
ys = dict(zip(ORDER, range(n)[::-1]))
MRG, MRG_LT, NOM = "#D62728", "#F7BCC3", "#BDBBB6"
GRAYTXT, REDTXT = "#6f6e69", "#B01E1E"
SPAN = 1.72

W, H = 2.94, 2.65
fig = plt.figure(figsize=(0.95 * W, 0.95 * H))
B, T = 0.48, 0.36
XA, XB, PW = 0.50, 1.62, 1.02
axA = fig.add_axes([XA / W, B / H, PW / W, (H - B - T) / H])
axB = fig.add_axes([XB / W, B / H, PW / W, (H - B - T) / H], sharey=axA)

def style(ax):
    ax.axvline(0, color="#c3c2b7", lw=1, zorder=1)
    ax.axhline(ys["mean"] - 0.5, color="#dedcd4", lw=0.8, ls=":", zorder=0)
    ax.axhline(ys["fin"] + 0.5, color="#dedcd4", lw=0.8, ls=":", zorder=0)
    ax.grid(axis="x", which="both", color="#eeede9", lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", labelsize=11, pad=2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(-0.6, n - 0.4)
    ax.patch.set_visible(False)

h = 0.45
for d in ORDER:
    v, y = R[d]["dmerge"], ys[d]
    axA.barh(y, v, height=h, color=MRG, alpha=0.55, lw=0, zorder=3)
    inside = v < -0.9
    axA.annotate(f"{v:+.2f}", (v, y), textcoords="offset points", xytext=(2 if inside else -2, 0),
                 ha="left" if inside else "right", va="center", fontsize=7.0, fontweight="bold",
                 color="white" if inside else GRAYTXT)
style(axA)
axA.set_xlim(0.30 - SPAN, 0.30); axA.set_xticks([-1.0, 0.0]); axA.set_xticks([-0.5], minor=True)
axA.set_xticklabels(["−1", "0"])
axA.set_yticks([ys[d] for d in ORDER]); axA.set_yticklabels([LABEL[d] for d in ORDER], fontsize=11)

for d in ORDER:
    r, y = R[d], ys[d]
    gm, gn = r["dmopd"] - r["dmerge"], r["dnomerge"]
    axB.barh(y + h / 2, gm, height=h, color=MRG, zorder=3)
    axB.barh(y - h / 2, gn, height=h, color=NOM, zorder=3)
    axB.annotate(f"{gm - gn:+.2f}", (max(gm, gn, 0), y + (h / 2 if max(gm, gn) > .95 else 0)), textcoords="offset points", xytext=(-2 if max(gm, gn) > .95 else 2, 0),
                 ha="right" if max(gm, gn) > .95 else "left", va="center", fontsize=7.0, fontweight="bold", color="white" if max(gm, gn) > .95 else (REDTXT if gm >= gn else GRAYTXT))
style(axB)
axB.set_xlim(-0.30, SPAN - 0.30); axB.set_xticks([0.0, 1.0]); axB.set_xticks([0.5], minor=True)
axB.set_xticklabels(["0", "1"])
axB.tick_params(axis="y", labelleft=False)

def entry(color, text):
    return HPacker(children=[TextArea("■", textprops=dict(color=color, fontsize=12)),
                             TextArea(text, textprops=dict(color="#2b2a28", fontsize=9))],
                   align="baseline", pad=0, sep=3)
key = HPacker(children=[entry(MRG, "With merge"), entry(NOM, "No merge")], align="baseline", pad=0, sep=12)
fig.add_artist(AnnotationBbox(key, (0.5, 2.61 / H), xycoords="figure fraction",
                              box_alignment=(0.5, 1.0), frameon=False))
yl = 0.14 / H
cA, cB = XA + PW / 2, XB + PW / 2
fig.text(cA / W, yl, "Merge", ha="center", va="center", fontsize=11, color="#2b2a28")
fig.text((cB + 0.10) / W, yl, "+50 MOPD steps", ha="center", va="center", fontsize=11, color="#2b2a28")
fig.add_artist(matplotlib.patches.FancyArrowPatch(((cA + 0.31) / W, yl), ((cB - 0.54) / W, yl),
               transform=fig.transFigure, arrowstyle="-|>", mutation_scale=9, lw=1.2, color="#6f6e69"))
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig3a.{ext}", dpi=300)
plt.close(fig)
