"""Generate a Nature-style, colour-blind-safe OXR service comparison chart.

Native MegaLM vs official vLLM 0.25 across four service scenarios. Uses the
Okabe–Ito colour-blind-safe palette and distinct markers / line styles so the
figure remains readable in greyscale and for all colour-vision types.

Usage::

    python docs/make_oxr_comparison_chart.py \
        --native /tmp/megalm-oxr-native-service.json \
        --vllm /tmp/megalm-oxr-vllm-service.json \
        --output docs/benchmarks/oxr_vllm_comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# Nature-style typography (clean sans-serif, no CJK required).
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.edgecolor": "#1A1A1A",
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    }
)

# Okabe–Ito palette: blue vs vermilion is the most robust dichromat-safe pair.
C_NAT = "#0072B2"
C_VLL = "#D55E00"
C_DARK = "#1A1A1A"
C_GREY = "#6B6B6B"
C_GRID = "#E6E6E6"


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.5)
    ax.xaxis.grid(False)
    ax.tick_params(direction="out", length=3.5, width=0.8, colors=C_DARK)


def panel_label(ax, letter):
    ax.text(
        -0.13, 1.10, letter, transform=ax.transAxes,
        fontsize=13, fontweight="bold", ha="left", va="bottom", color=C_DARK,
    )


def legend_nature(ax, loc="best"):
    return ax.legend(
        loc=loc, frameon=False, handlelength=2.0,
        handletextpad=0.6, borderpad=0.2, labelspacing=0.3,
    )


def build(native, vllm, output):
    tokens = [16, 32, 64, 128]
    concur = [1, 2, 4, 8]

    nat_d = [native["single"][str(t)]["decode_tok_s"] for t in tokens]
    vll_d = [vllm["single"][str(t)]["decode_tok_s"] for t in tokens]
    ttft_nat = native["single"]["16"]["ttft_ms"]
    ttft_vll = vllm["single"]["16"]["ttft_ms"]
    nat_c = [native["concurrent"][str(n)]["aggregate_tok_s"] for n in concur]
    vll_c = [vllm["concurrent"][str(n)]["aggregate_tok_s"] for n in concur]
    nat_e = [native["single"][str(t)]["total_s"] * 1000 for t in tokens]
    vll_e = [vllm["single"][str(t)]["total_s"] * 1000 for t in tokens]

    fig = plt.figure(figsize=(13.5, 8.2))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        2, 2, hspace=0.42, wspace=0.26, left=0.075, right=0.975, top=0.88, bottom=0.11
    )

    # a) Single-stream decode throughput
    ax1 = fig.add_subplot(gs[0, 0])
    style(ax1)
    x = np.arange(len(tokens))
    w = 0.36
    ax1.bar(x - w / 2, nat_d, w, label="MegaLM (native)", color=C_NAT, edgecolor="none")
    ax1.bar(x + w / 2, vll_d, w, label="vLLM 0.25", color=C_VLL, edgecolor="none")
    for bars in (ax1.patches[: len(tokens)], ax1.patches[len(tokens) :]):
        for bar in bars:
            ax1.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                f"{bar.get_height():.0f}", ha="center", va="bottom",
                fontsize=7.5, color=C_DARK,
            )
    ax1.set_xticks(x)
    ax1.set_xticklabels(tokens)
    ax1.set_xlabel("Output length (tokens)")
    ax1.set_ylabel(r"Decode throughput (tokens s$^{-1}$)")
    ax1.set_ylim(0, max(nat_d) * 1.30)
    legend_nature(ax1, loc="upper left")
    panel_label(ax1, "a")

    # b) TTFT
    ax2 = fig.add_subplot(gs[0, 1])
    style(ax2)
    bars = ax2.bar(
        [0, 1], [ttft_nat, ttft_vll], width=0.5,
        color=[C_NAT, C_VLL], edgecolor="none",
    )
    for bar, v in zip(bars, [ttft_nat, ttft_vll]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.18,
            f"{v:.2f} ms", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=C_DARK,
        )
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["MegaLM (native)", "vLLM 0.25"])
    ax2.set_ylabel("Time to first token (ms)")
    ax2.set_ylim(0, max(ttft_nat, ttft_vll) * 1.35)
    ax2.annotate(
        f"{ttft_vll / ttft_nat:.2f}×",
        xy=(0, ttft_nat + 0.1), xytext=(0.55, ttft_vll * 0.72),
        ha="center", fontsize=10, fontweight="bold", color=C_NAT,
        arrowprops=dict(
            arrowstyle="-|>", color=C_NAT, lw=1.2,
            connectionstyle="arc3,rad=-0.2",
        ),
    )
    panel_label(ax2, "b")

    # c) Concurrent aggregate throughput
    ax3 = fig.add_subplot(gs[1, 0])
    style(ax3)
    x = np.arange(len(concur))
    ax3.plot(
        x, nat_c, "-o", color=C_NAT, linewidth=1.8, markersize=6,
        markeredgecolor="white", markeredgewidth=0.6, label="MegaLM (native)",
    )
    ax3.plot(
        x, vll_c, "--s", color=C_VLL, linewidth=1.8, markersize=6,
        markeredgecolor="white", markeredgewidth=0.6, label="vLLM 0.25",
    )
    for i, (nv, vv) in enumerate(zip(nat_c, vll_c)):
        ax3.annotate(
            f"+{(nv / vv - 1) * 100:.0f}%", (i, nv),
            textcoords="offset points", xytext=(0, 9),
            ha="center", fontsize=8, fontweight="bold", color=C_NAT,
        )
    ax3.set_xticks(x)
    ax3.set_xticklabels(concur)
    ax3.set_xlabel("Concurrent requests")
    ax3.set_ylabel(r"Aggregate output throughput (tokens s$^{-1}$)")
    ax3.set_ylim(0, max(nat_c) * 1.18)
    ax3.yaxis.set_major_formatter(
        FuncFormatter(lambda v, p: f"{v/1000:.1f}k" if v >= 1000 else f"{int(v)}")
    )
    legend_nature(ax3, loc="upper left")
    panel_label(ax3, "c")

    # d) E2E latency
    ax4 = fig.add_subplot(gs[1, 1])
    style(ax4)
    x = np.arange(len(tokens))
    ax4.plot(
        x, nat_e, "-o", color=C_NAT, linewidth=1.8, markersize=6,
        markeredgecolor="white", markeredgewidth=0.6, label="MegaLM (native)",
    )
    ax4.plot(
        x, vll_e, "--s", color=C_VLL, linewidth=1.8, markersize=6,
        markeredgecolor="white", markeredgewidth=0.6, label="vLLM 0.25",
    )
    for i, (nv, vv) in enumerate(zip(nat_e, vll_e)):
        ax4.annotate(
            f"−{(1 - nv / vv) * 100:.0f}%", (i, nv),
            textcoords="offset points", xytext=(0, -16),
            ha="center", fontsize=8, fontweight="bold", color=C_NAT,
        )
    ax4.set_xticks(x)
    ax4.set_xticklabels(tokens)
    ax4.set_xlabel("Output length (tokens)")
    ax4.set_ylabel("End-to-end latency (ms)")
    ax4.set_ylim(0, max(vll_e) * 1.12)
    legend_nature(ax4, loc="upper left")
    panel_label(ax4, "d")

    fig.suptitle(
        "OXR-1.0 Inference Serving: Native MegaLM vs Official vLLM 0.25",
        fontsize=13, fontweight="bold", x=0.5, y=0.965, color=C_DARK,
    )
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    print("saved", output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", default="/tmp/megalm-oxr-native-service.json")
    parser.add_argument("--vllm", default="/tmp/megalm-oxr-vllm-service.json")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("oxr_vllm_comparison.png")),
    )
    args = parser.parse_args()
    build(json.load(open(args.native)), json.load(open(args.vllm)), args.output)


if __name__ == "__main__":
    main()
