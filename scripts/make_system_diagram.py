"""Generate the system diagram (Figure 1) for the paper.

Closed loop: dynamical prior -> GP residual correction -> robot acquisition
-> assimilation -> next prediction.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np


def main():
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(-0.7, 14.2)
    ax.set_ylim(-0.3, 7.5)
    ax.axis("off")

    # Box style
    def box(x, y, w, h, label, color, sub=None, fontsize=12):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                          fc=color, ec="black", linewidth=1.5, alpha=0.9)
        ax.add_patch(b)
        if sub:
            ax.text(x + w / 2, y + h * 0.62, label,
                   ha="center", va="center", fontsize=fontsize, fontweight="bold")
            ax.text(x + w / 2, y + h * 0.30, sub,
                   ha="center", va="center", fontsize=fontsize - 3, style="italic")
        else:
            ax.text(x + w / 2, y + h / 2, label,
                   ha="center", va="center", fontsize=fontsize, fontweight="bold")

    def arrow(x1, y1, x2, y2, label="", color="black", offset=0.15, label_above=True):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle="-|>", mutation_scale=20,
                           color=color, linewidth=2)
        ax.add_patch(a)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dy = offset if label_above else -offset
            ax.text(mx, my + dy, label, ha="center", va="center",
                   fontsize=10, color=color, fontweight="bold",
                   bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.85))

    # ── Top loop: dynamical model + correction ──
    # Box 1: Dynamical prior (FNO or persistence)
    box(0.5, 4.5, 2.7, 1.5, "Dynamical prior",
       sub="FNO  /  Persistence", color="#cce4f6")

    # Box 2: GP residual correction
    box(5.0, 4.5, 2.7, 1.5, "GP residual",
       sub="correction", color="#ffd9b3")

    # Box 3: Corrected estimate
    box(9.5, 4.5, 3.5, 1.5, "Corrected estimate",
       sub=r"$\hat{y}=\hat{y}_{prior}+\mu_{GP}$", color="#d6f5d6")

    # Arrows top row
    arrow(3.2, 5.25, 5.0, 5.25, label=r"$\hat{y}_{prior}$")
    arrow(7.7, 5.25, 9.5, 5.25, label=r"$\mu_{GP},\sigma_{GP}$")

    # ── Bottom loop: robot decisions and observations ──
    # Box 4: Robot acquisition / lookahead
    box(9.0, 1.0, 4.0, 1.5, "Multi-robot acquisition",
       sub="Voronoi + uncertainty / MI / UCB", color="#f5d0d0")

    # Box 5: Observations
    box(4.5, 1.0, 3.7, 1.5, "Robot observations",
       sub=r"$y_{true}(x_i)$ at $i=1...N$", color="#e8d5f0")

    # Box 6: Residual computation
    box(0.0, 1.0, 3.5, 1.5, "Residual update",
       sub=r"$e_i = y_{true}(x_i) - \hat{y}_{prior}(x_i)$", color="#ffd9b3")

    # Bottom-row arrows
    arrow(11.0, 4.5, 11.0, 2.5, label="plan next\nsample sites", offset=0.4, label_above=False)
    arrow(9.0, 1.75, 8.2, 1.75, label="execute")
    arrow(4.5, 1.75, 3.5, 1.75, label="add to GP")

    # Closing the loop: residual update -> GP correction
    arrow(1.75, 2.5, 1.75, 4.5, label="next step")

    # Legend / arrows for assimilation cycle
    ax.text(7.0, 7.1, "Per assimilation step (every K=1 forecast steps)",
           ha="center", fontsize=11, fontweight="bold", color="#444")

    plt.tight_layout()
    out_path = os.path.join(ROOT, "results", "dynamic_ipp", "final", "system_diagram.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
