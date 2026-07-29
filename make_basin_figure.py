# -*- coding: utf-8 -*-
"""Generate basin schematic for paper."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

FIG = Path(__file__).parent / "figures" / "basin_schematic.png"
FIG.parent.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")

stations = [
    (1.5, 3.5, "01636500\nShenandoah R.\n(Millville, WV)", "#40916C"),
    (5, 3.5, "01638480\nPotomac R.\n(Harpers Ferry)", "#2D6A4F"),
    (8.5, 2, "01646500\nPotomac R.\n(Washington, DC)", "#1B4332"),
]

for x, y, label, color in stations:
    box = FancyBboxPatch((x - 0.9, y - 0.7), 1.8, 1.4, boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor="white", alpha=0.9)
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=8, color="white", fontweight="bold")

ax.annotate("", xy=(7.6, 2.3), xytext=(5.9, 3.2),
            arrowprops=dict(arrowstyle="->", color="#555", lw=2))
ax.annotate("", xy=(4.1, 3.5), xytext=(2.4, 3.5),
            arrowprops=dict(arrowstyle="->", color="#555", lw=2))
ax.text(5, 0.5, "Flow direction →", ha="center", fontsize=11, color="#333")
ax.set_title("Potomac River Basin Monitoring Network", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG, dpi=200, bbox_inches="tight")
print(f"Saved {FIG}")
