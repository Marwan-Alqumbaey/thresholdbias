"""Figures 1 to 4 of the paper (PDF, 1000-dpi TIFF and 300-dpi PNG)."""

import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image
from thresholdbias import exact_conditional

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures"
BLUE, ORANGE, AQUA, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.edgecolor": "#9c9b96",
        "axes.linewidth": 0.7,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e1",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
    }
)
WIDTH = 6.69  # 17 cm


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=1000, bbox_inches="tight")
    Image.open(buf).convert("RGB").save(OUT / f"{name}.tif", dpi=(1000, 1000), compression="tiff_lzw")
    plt.close(fig)


def figure1():
    ps = np.linspace(0.005, 0.10, 191)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7), sharey=True)
    for ax, rr, ttl in [(axes[0], 1.0, "A   True risk ratio 1"), (axes[1], 1.5, "B   True risk ratio 1.5")]:
        for (n1, n0), col, lab in [
            ((150, 150), BLUE, "1:1 (150 vs 150)"),
            ((200, 100), ORANGE, "2:1 (200 vs 100)"),
            ((225, 75), AQUA, "3:1 (225 vs 75)"),
        ]:
            y = [exact_conditional(n1, n0, min(rr * p, 0.99), p, 0.05)["rr_cond"] for p in ps]
            ax.plot(ps * 100, y, color=col, label=lab)
        ax.axhline(rr, color=GREY, lw=0.9, ls="--")
        ax.axvline(5, color=GREY, lw=0.7, ls=":")
        ax.text(5.15, 0.215, "5% threshold", color=INK2, fontsize=8)
        ax.set_title(ttl)
        ax.set_xlabel("Placebo event rate (%)")
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.set_yticks([0.25, 0.5, 1, 1.5, 2, 4])
        ax.set_yticklabels(["0.25", "0.5", "1", "1.5", "2", "4"])
    axes[0].set_ylabel("Posted-row risk ratio")
    axes[0].set_ylim(0.2, 4.5)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Allocation (drug:placebo)",
        title_fontsize=8,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.12),
    )
    fig.tight_layout()
    save(fig, "Figure1")


def figure2():
    D = pd.read_csv(RES / "simulation" / "naive_methods.csv")
    x = D[(D.K == 30) & (D.N == 300) & (D.c == 0.05)]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.0), sharex=True)
    cols = {"1:1": BLUE, "2:1": ORANGE, "3:1": AQUA}
    mark = {"1:1": "o", "2:1": "s", "3:1": "^"}
    for j, rr in enumerate([1.0, 1.5]):
        for ratio, col in cols.items():
            for meth, ls, lw in [("M1", "-", 1.6), ("M7", ":", 1.4)]:
                d = x[(x.rr == rr) & (x.ratio == ratio) & (x.method == meth)].sort_values("pmed")
                ok = d.n >= 500
                for yv, ax in [("bias", axes[0, j]), ("cover", axes[1, j])]:
                    ax.plot(d.pmed[ok] * 100, d[yv][ok], ls=ls, color=col, lw=lw, marker=mark[ratio], ms=3.6)
                    ax.plot(d.pmed[~ok] * 100, d[yv][~ok], ls="none", marker=mark[ratio], ms=4, mfc="white", mec=col)
        axes[0, j].axhline(0, color=GREY, lw=0.9, ls="--")
        axes[1, j].axhline(0.95, color=GREY, lw=0.9, ls="--")
        axes[0, j].set_title(f"{'AB'[j]}   Bias, true risk ratio {rr:g}")
        axes[1, j].set_title(f"{'CD'[j]}   Coverage, true risk ratio {rr:g}")
        axes[1, j].set_xlabel("Median placebo event rate (%)")
        axes[1, j].set_ylim(0.5, 1.02)
    axes[0, 0].set_ylabel("Bias in log risk ratio")
    axes[1, 0].set_ylabel("Coverage of 95% CI")
    h = [
        Line2D([], [], color=c, lw=1.6, marker=mark[r], ms=3.6, label=f"{r} drug:placebo") for r, c in cols.items()
    ] + [
        Line2D([], [], color=INK2, lw=1.6, ls="-", label="Any-arm rule"),
        Line2D([], [], color=INK2, lw=1.4, ls=":", label="Pooled-arm rule"),
        Line2D(
            [], [], color=INK2, marker="o", mfc="white", ls="none", label="Open symbol: <50% of replicates estimable"
        ),
    ]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "Figure2")


def figure3():
    R = read_json(RES / "simulation" / "corrected_methods.json")
    R = sorted([r for r in R if r["K"] == 30], key=lambda r: (r["rr"], r["ratio"], r["pmed"]))
    cols = {"M0": GREY, "M1": ORANGE, "M5": BLUE, "M6": AQUA}
    mark = {"M0": "o", "M1": "s", "M5": "^", "M6": "D"}
    labs = {
        "M0": "M0: complete data (reference)",
        "M1": "M1: posted rows, Mantel-Haenszel",
        "M5": "M5: corrected likelihood, posted rows",
        "M6": "M6: full likelihood, with unposted trials",
    }
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 4.6), sharey=True)
    ylab = [f"RR {r['rr']:g}, {r['ratio']}, {r['pmed'] * 100:g}%" for r in R]
    for m, dy in zip(cols, [0.27, 0.09, -0.09, -0.27]):
        for i, r in enumerate(R):
            v = r["results"][m]
            y = len(R) - 1 - i + dy
            axes[0].errorbar(
                v["bias"],
                y,
                xerr=1.96 * v["mcse_bias"],
                fmt=mark[m],
                ms=3.2,
                color=cols[m],
                elinewidth=0.9,
                label=labs[m] if i == 0 else None,
            )
            axes[1].plot(v["cover"], y, mark[m], ms=3.2, color=cols[m])
    axes[0].set_yticks(range(len(R)))
    axes[0].set_yticklabels(ylab[::-1], fontsize=8)
    axes[0].axvline(0, color=GREY, lw=0.9, ls="--")
    axes[1].axvline(0.95, color=GREY, lw=0.9, ls="--")
    axes[0].set_xlabel("Bias in log risk ratio (±1.96 Monte Carlo SE)")
    axes[1].set_xlabel("Coverage of 95% CI")
    axes[0].set_title("A   Bias")
    axes[1].set_title("B   Coverage")
    axes[1].set_xlim(0.6, 1.0)
    for a in axes:
        a.grid(axis="y", visible=False)
    fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "Figure3")


def figure4():
    S = read_json(RES / "registry" / "rethreshold_primary.json")
    S20 = read_json(RES / "registry" / "rethreshold_minarm20.json")
    Ssm = read_json(RES / "registry" / "rethreshold_smallarm20.json")
    P = pd.read_csv(RES / "registry" / "predicted_shift_primary_terms.csv")
    fig = plt.figure(figsize=(WIDTH, 5.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.08], hspace=0.42, wspace=0.62)
    ax = fig.add_subplot(gs[0, :])
    strata = ["<0.9", "0.9-1.1", "1.1-1.7", ">=1.7"]
    lab = [
        "Below 0.9\n(placebo arm larger)",
        "0.9 to 1.1\n(near-equal)",
        "1.1 to 1.7",
        "1.7 or more\n(drug arm larger)",
    ]
    for i, s in enumerate(strata):
        for dx, key, col, mk, nm in [
            (-0.11, "post", ORANGE, "o", "Any-arm rule (current)"),
            (0.11, "pool", BLUE, "s", "Pooled-arm rule"),
        ]:
            m = S[s][f"median_shift_{key}"]
            lo, hi = S[s][f"ci_{key}"]
            ax.plot([i + dx] * 2, [lo, hi], color=col, lw=1.6)
            ax.plot(i + dx, m, mk, color=col, ms=5.5, mec="white", mew=1.0, label=nm if i == 0 else None)
        ax.text(i, -0.137, f"{S[s]['estimable_post']} terms", ha="center", color=INK2, fontsize=8)
    ax.axhline(0, color=GREY, lw=0.9, ls="--")
    ax.set_xticks(range(4))
    ax.set_xticklabels(lab)
    ax.set_xlabel("Allocation ratio (drug:placebo participants at risk)")
    ax.set_ylim(-0.15, 0.11)
    ax.set_ylabel("Median shift in log risk ratio")
    ax.set_title("A   Shift by allocation ratio")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    ax = fig.add_subplot(gs[1, 0])
    E = S[">=1.7"]
    rows = [
        ("All terms (median)", E["median"]),
        ("Precision-weighted mean", E["ivw_mean"]),
        ("Terms posted by ≥5 trials", E["median_ge5rows"]),
        ("Terms posted by <5 trials", E["median_lt5rows"]),
        ("Terms with risk ratio near 1", E["median_nearnull"]),
        ("An arm of <20 participants", dict(est=Ssm[">=1.7"]["median_shift_post"], ci=Ssm[">=1.7"]["ci_post"])),
        ("Both arms ≥20 participants", dict(est=S20[">=1.7"]["median_shift_post"], ci=S20[">=1.7"]["ci_post"])),
    ]
    for i, (nm, d) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot(d["ci"], [y, y], color=ORANGE, lw=1.6)
        ax.plot(d["est"], y, "o", color=ORANGE, ms=5, mec="white", mew=1.0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.axvline(0, color=GREY, lw=0.9, ls="--")
    ax.set_xlabel("Shift in log risk ratio (95% CI)")
    ax.set_title("B   Drug arm ≥1.7 × placebo arm")
    ax.grid(axis="y", visible=False)
    ax = fig.add_subplot(gs[1, 1])
    g = P[P.stratum == ">=1.7"].dropna(subset=["predicted_shift"])
    ax.scatter(g.predicted_shift, g.shift_post.clip(-3, 3), s=7, color=ORANGE, alpha=0.55, lw=0)
    lim = [-2.5, 1.5]
    ax.plot(lim, lim, color=GREY, lw=0.9, ls="--")
    b = np.polyfit(g.predicted_shift, g.shift_post, 1)
    xx = np.array(lim)
    ax.plot(xx, b[1] + b[0] * xx, color=INK, lw=1.2)
    rho = g[["predicted_shift", "shift_post"]].corr("spearman").iloc[0, 1]
    ax.text(0.97, 0.04, f"Spearman ρ = {rho:.2f}", transform=ax.transAxes, fontsize=8, color=INK2, ha="right")
    ax.set_xlim(lim)
    ax.set_ylim(-3.1, 1.6)
    ax.set_xlabel("Predicted shift")
    ax.set_ylabel("Observed shift")
    ax.set_title("C   Observed vs predicted")
    save(fig, "Figure4")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    figure1()
    figure2()
    figure3()
    figure4()
