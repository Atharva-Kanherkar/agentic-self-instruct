"""Generate charts for the LegalBench hearsay DataSmith-vs-control experiment.

Run:  uv run --with matplotlib --with numpy python docs/benchmarks/make_hearsay_charts.py
Writes PNGs into docs/benchmarks/assets/.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "assets"
OUT.mkdir(parents=True, exist_ok=True)

# ---- results: LegalBench hearsay, Qwen2.5-7B, 72 verified pairs/arm, 3 seeds ----
ARMS = ["Baseline\n(no training)", "Naive control\n(72 pairs)", "DataSmith\n(72 pairs)"]
COLORS = {"baseline": "#9aa0a6", "control": "#e8833a", "datasmith": "#3a7bd5"}
bal_mean = [69.5, 71.7, 73.1]          # balanced accuracy %
bal_lo   = [69.5, 71.0, 73.1]
bal_hi   = [69.5, 72.0, 73.1]
yes_recall = [0.805, 0.610, 0.707]
no_recall  = [0.585, 0.824, 0.755]
# prediction distributions (Yes, No) out of 94; true = 41 Yes / 53 No
preds = {
    "True labels":     (41, 53),
    "Baseline":        (55, 39),
    "Naive control":   (35, 59),
    "DataSmith":       (42, 52),
}
MAJORITY = 56.4

# ============================ 1. balanced accuracy ============================
fig, ax = plt.subplots(figsize=(7.2, 4.6))
x = np.arange(len(ARMS))
cols = [COLORS["baseline"], COLORS["control"], COLORS["datasmith"]]
err = [[m - lo for m, lo in zip(bal_mean, bal_lo)], [hi - m for m, hi in zip(bal_mean, bal_hi)]]
bars = ax.bar(x, bal_mean, yerr=err, capsize=6, color=cols, edgecolor="black", linewidth=0.6, width=0.6)
ax.axhline(MAJORITY, ls="--", color="#c0392b", lw=1.2)
ax.text(2.45, MAJORITY + 0.3, "majority-class floor 56.4%", color="#c0392b", ha="right", fontsize=9)
for b, m in zip(bars, bal_mean):
    ax.text(b.get_x() + b.get_width()/2, m + 0.5, f"{m:.1f}%", ha="center", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(ARMS)
ax.set_ylabel("Balanced accuracy (%)")
ax.set_ylim(54, 78)
ax.set_title("LegalBench hearsay — balanced accuracy (Qwen2.5-7B, 3 seeds)", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "fig1_balanced_accuracy.png", dpi=150); plt.close(fig)

# ===================== 2. yes vs no recall (calibration) =====================
fig, ax = plt.subplots(figsize=(7.6, 4.6))
labels = ["Baseline", "Naive control", "DataSmith"]
x = np.arange(len(labels)); w = 0.36
b1 = ax.bar(x - w/2, yes_recall, w, label="Yes-recall (catches hearsay)", color="#5b8def", edgecolor="black", lw=0.5)
b2 = ax.bar(x + w/2, no_recall, w, label="No-recall (spots non-hearsay)", color="#f2a65a", edgecolor="black", lw=0.5)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.012, f"{b.get_height():.2f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Recall"); ax.set_ylim(0, 1.0)
ax.set_title("Per-class recall: DataSmith stays calibrated, control over-corrects", fontweight="bold")
ax.legend(frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.28))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "fig2_recall_breakdown.png", dpi=150); plt.close(fig)

# ===================== 3. prediction distribution pies =======================
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
pie_cols = ["#5b8def", "#f2a65a"]  # Yes, No
for ax, (name, (y, n)) in zip(axes, preds.items()):
    ax.pie([y, n], labels=["Yes", "No"], colors=pie_cols, autopct=lambda p: f"{p*0.94:.0f}",
           startangle=90, wedgeprops=dict(edgecolor="white", linewidth=1.5), textprops={"fontsize": 10})
    ax.set_title(name, fontweight="bold", fontsize=11)
fig.suptitle("Predicted label mix on the 94-example test set  (counts; true = 41 Yes / 53 No)",
             fontweight="bold", y=1.04)
fig.tight_layout(); fig.savefig(OUT / "fig3_prediction_pies.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# =============== 4. data generation: gate funnel + label dist ================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 3.8))
# funnel (representative per-batch rates: candidates -> strong correct -> kept pair)
stages = ["Candidates\ngenerated", "Strong solver\ncorrect (~93%)", "Kept pair\n(weak wrong +\nstrong right)"]
vals = [100, 93, 32]
axL.barh(range(len(stages))[::-1], vals, color=["#bdc3c7", "#3a7bd5", "#27ae60"], edgecolor="black", lw=0.6)
for i, v in zip(range(len(stages))[::-1], vals):
    axL.text(v + 1.5, i, f"{v}%", va="center", fontweight="bold")
axL.set_yticks(range(len(stages))[::-1]); axL.set_yticklabels(stages)
axL.set_xlim(0, 110); axL.set_xlabel("% of candidates")
axL.set_title("Weak-vs-strong + correctness gate", fontweight="bold")
axL.spines[["top", "right"]].set_visible(False)
# label dist of kept DataSmith pairs (all No — one-directional failure frontier)
axR.pie([0.001, 100], labels=["Yes (0)", "No (72)"], colors=["#5b8def", "#f2a65a"],
        startangle=90, wedgeprops=dict(edgecolor="white", linewidth=1.5), textprops={"fontsize": 11})
axR.set_title("DataSmith verified pairs:\n100% 'No' (one-directional frontier)", fontweight="bold", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "fig4_generation.png", dpi=150); plt.close(fig)

print("wrote:", *(p.name for p in sorted(OUT.glob("*.png"))))
