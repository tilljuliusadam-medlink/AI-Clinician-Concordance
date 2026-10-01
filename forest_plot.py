"""
Figure 1: forest plot of the F1 main block, drawn from the two regression tables in results/.
"""

import re

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config as cfg  # noqa: E402

BLOCK = "f1_main"
PANELS = [("table2_binary_regressions.csv", "OR (95% CI)", "Odds ratio (log scale, 95% CI)", 1.0, True,
           "12-month binary outcomes"),
          ("table3_continuous_regressions.csv", "Standardized beta (95% CI)", "Standardized beta (95% CI)", 0.0,
           False, "12-month continuous outcomes")]


def parse(cell):
    """'0.45 (0.35, 0.57)' -> (0.45, 0.35, 0.57)."""
    return tuple(float(x) for x in re.fullmatch(r"(\S+) \((\S+), (\S+)\)", cell).groups())


def significant(p_bh):
    return p_bh == "<0.001" or float(p_bh) < 0.05


def draw(ax, table, column, xlabel, null, log, title):
    rows = table[table["Block"] == BLOCK].reset_index(drop=True)
    labels, y = [], 0
    for domain, group in rows.groupby("Domain", sort=False):
        labels.append((y, domain, True))
        y += 1
        for _, row in group.iterrows():
            point, lo, hi = parse(row[column])
            filled = significant(row["p (BH)"])
            ax.plot([lo, hi], [y, y], color="black", linewidth=1)
            ax.plot(point, y, "o", color="black", markerfacecolor="black" if filled else "white", markersize=5)
            labels.append((y, f"{row['Outcome']}   {row[column]}", False))
            y += 1
    ax.axvline(null, linestyle="--", color="grey", linewidth=0.8)
    if log:
        ax.set_xscale("log")
    ax.set_yticks([pos for pos, _, _ in labels])
    ax.set_yticklabels([text for _, text, _ in labels])
    for tick, (_, _, heading) in zip(ax.get_yticklabels(), labels):
        tick.set_fontweight("bold" if heading else "normal")
    ax.set_ylim(y - 0.5, -0.5)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold")


def main():
    fig, axes = plt.subplots(2, 1, figsize=(10, 13), gridspec_kw={"height_ratios": [25, 16]})
    for ax, (filename, column, xlabel, null, log, title) in zip(axes, PANELS):
        draw(ax, pd.read_csv(cfg.RESULTS_DIR / filename, dtype=str), column, xlabel, null, log, title)
    fig.tight_layout()
    fig.savefig(cfg.RESULTS_DIR / "figure1_forest_plot.png", dpi=150)
    print("[write] figure1_forest_plot.png")


if __name__ == "__main__":
    main()
