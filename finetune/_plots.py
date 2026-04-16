import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from finetune._data import COLORS, DISEASE_ORDER, METHOD_DISPLAY, PLOT_RC

plt.rcParams.update(PLOT_RC)


def plot_mean_auc(
    all_results: dict,
    classes: list[str],
    ns: list[int],
    mname: str,
    epoch: int,
    out_path: Path,
    title_prefix: str = "",
) -> None:
    display_name = METHOD_DISPLAY.get(mname, mname)
    fig, ax = plt.subplots(figsize=(7, 5))
    for init_name in all_results:
        means = [np.mean([all_results[init_name][d][n]["auc"][0] for d in classes]) for n in ns]
        stds  = [np.mean([all_results[init_name][d][n]["auc"][1] for d in classes]) for n in ns]
        ax.plot(ns, means, marker="o", label=init_name, color=COLORS[init_name])
        ax.fill_between(
            ns,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.15, color=COLORS[init_name],
        )
    label = f"{title_prefix.lower()} " if title_prefix else ""
    ax.set_xlabel("Shots per disease")
    ax.set_ylabel("Mean AUC (binary, across diseases)")
    ax.set_title(f"{display_name} {label}mean AUC\n{len(classes)} diseases")
    ax.legend()
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.close()


def plot_per_disease(
    all_results: dict,
    classes: list[str],
    ns: list[int],
    mname: str,
    epoch: int,
    out_path: Path,
    title_prefix: str = "",
    secondary_results: dict | None = None,
) -> None:
    display_name = METHOD_DISPLAY.get(mname, mname)

    # Sort by canonical DISEASE_ORDER, append any unknowns
    ordered = [d for d in DISEASE_ORDER if d in classes]
    ordered += [d for d in classes if d not in ordered]

    n_cols = 3
    n_full_rows = len(ordered) // n_cols
    remainder = len(ordered) % n_cols
    n_rows = n_full_rows + (1 if remainder else 0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # For a single leftover disease (remainder=1) center it in the last row
    if remainder == 1:
        axes[n_rows - 1, 0].set_visible(False)
        axes[n_rows - 1, 2].set_visible(False)
    elif remainder == 2:
        axes[n_rows - 1, 2].set_visible(False)

    # Build ordered list of active axes
    active_axes: list = []
    for r in range(n_rows - 1 if remainder else n_rows):
        for c in range(n_cols):
            active_axes.append(axes[r, c])
    if remainder == 1:
        active_axes.append(axes[n_rows - 1, 1])
    elif remainder == 2:
        active_axes.extend([axes[n_rows - 1, 0], axes[n_rows - 1, 1]])

    for i, disease in enumerate(ordered):
        ax = active_axes[i]

        # Primary results: solid lines
        for init_name in all_results:
            means = [all_results[init_name][disease][n]["auc"][0] for n in ns]
            stds  = [all_results[init_name][disease][n]["auc"][1] for n in ns]
            ax.plot(ns, means, marker="o", color=COLORS[init_name], linewidth=2.0, linestyle="-")
            ax.fill_between(
                ns,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.10, color=COLORS[init_name],
            )

        # Secondary results (finetune): dashed lines
        if secondary_results is not None:
            for init_name, diseases_data in secondary_results.items():
                if disease not in diseases_data:
                    continue
                means = [diseases_data[disease][n]["auc"][0] for n in ns]
                ax.plot(ns, means, marker="s", color=COLORS[init_name], linewidth=2.0,
                        linestyle="--", alpha=0.8, markersize=5)

        ax.axhline(0.5, color="#aaaaaa", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_title(disease.upper(), fontsize=14)
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels(ns, fontsize=12)
        ax.set_ylim(0.3, 1.05)
        ax.set_ylabel("AUC", fontsize=13)

        # Major grid + minor grid at every 0.1
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.grid(True, which="major", alpha=0.25, linewidth=0.8)
        ax.grid(True, which="minor", alpha=0.12, linewidth=0.4)

    # Legend: colors for init method + line styles for probe vs finetune
    color_handles = [
        Line2D([0], [0], color=COLORS["SSL pretrained"], lw=2, label="SSL pretrained"),
        Line2D([0], [0], color=COLORS["ImageNet"],       lw=2, label="ImageNet"),
        Line2D([0], [0], color=COLORS["Random init"],    lw=2, label="Random init"),
    ]
    if secondary_results is not None:
        primary_lbl = title_prefix.lower() if title_prefix else "probe"
        style_handles = [
            Line2D([0], [0], color="#888888", lw=2, ls="-",  label=primary_lbl),
            Line2D([0], [0], color="#888888", lw=2, ls="--", label="finetune"),
        ]
        legend_handles = color_handles + style_handles
        ncols_legend = 2
        title_str = f"{display_name} AUC per disease"
    else:
        legend_handles = color_handles
        ncols_legend = 1
        lbl = f"{title_prefix.lower()} " if title_prefix else ""
        title_str = f"{display_name} {lbl}AUC per disease"

    fig.legend(handles=legend_handles, loc="lower right", fontsize=13, ncol=ncols_legend)
    fig.suptitle(title_str, fontsize=20, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved per-disease plot to {out_path}")
    plt.close()
