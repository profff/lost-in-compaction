#!python3
"""
Generate schematic before/after figures for the four compaction strategies
described in §6.1 of LOST_IN_COMPACTION.md.

Replaces the ASCII art diagrams with proper PNG figures suitable for the
PDF / arXiv output.

Outputs:
    figures/fig_strategy_brutal.png
    figures/fig_strategy_incremental.png
    figures/fig_strategy_frozen.png
    figures/fig_strategy_frozenranked.png
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = Path(__file__).parent

# Style palette
C_RAW       = "#cfd8dc"   # raw messages — light grey
C_RECENT    = "#90caf9"   # recent / kept — light blue
C_SUMMARY   = "#fff59d"   # in-progress summary — yellow
C_FROZEN    = "#a5d6a7"   # frozen — green
C_FROZEN_R2 = "#66bb6a"   # frozen rank 2 — darker green
C_FROZEN_R3 = "#388e3c"   # frozen rank 3 — darkest green
C_EMPTY     = "#fafafa"
C_HIGH_WM   = "#e53935"   # red dashed
C_LOW_WM    = "#1e88e5"   # blue dashed
C_BUDGET    = "#8e24aa"   # purple dotted

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def draw_panel(ax, title, blocks, y_max=200, watermarks=None, budget=None,
               wm_side="left"):
    """blocks: list of (y_lo, y_hi, color, label, [callout]).
    Coordinates in kTok (0..200). If a 5th element 'callout' is True, the
    label is drawn outside the block with a leader line (used for thin slices).
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, y_max)
    ax.set_xticks([])
    ax.set_yticks([0, 60, 120, 180, 200])
    ax.set_yticklabels(["0K", "60K", "120K", "180K", "200K"], fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    # blocks
    for blk in blocks:
        if len(blk) == 4:
            y_lo, y_hi, color, label = blk
            callout = False
        else:
            y_lo, y_hi, color, label, callout = blk
        ax.add_patch(mpatches.Rectangle((0.05, y_lo), 0.9, y_hi - y_lo,
                                        facecolor=color, edgecolor="black",
                                        linewidth=0.6))
        if not label:
            continue
        if callout or (y_hi - y_lo) < 8:
            mid = (y_lo + y_hi) / 2
            cx, cy = 1.18, mid + 6 if mid < y_max - 12 else mid - 6
            ax.annotate(label, xy=(0.95, mid), xytext=(cx, cy),
                        fontsize=7.5, ha="left", va="center",
                        annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", lw=0.6, color="#666"))
        else:
            ax.text(0.5, (y_lo + y_hi) / 2, label, ha="center", va="center",
                    fontsize=7.5)
    # watermarks
    if wm_side == "left":
        x_lbl, ha = 0.07, "left"
    else:
        x_lbl, ha = 0.93, "right"
    if watermarks:
        for y, color, lbl in watermarks:
            ax.axhline(y, color=color, linestyle="--", linewidth=1.0)
            ax.text(x_lbl, y + 2, lbl, ha=ha, va="bottom",
                    fontsize=7, color=color, fontweight="bold")
    if budget is not None:
        y, lbl = budget
        ax.axhline(y, color=C_BUDGET, linestyle=":", linewidth=1.0)
        ax.text(x_lbl, y + 2, lbl, ha=ha, va="bottom",
                fontsize=7, color=C_BUDGET, fontweight="bold")


def add_arrow(fig, x0, y0, x1, y1, label=None):
    fig.patches.append(
        mpatches.FancyArrowPatch((x0, y0), (x1, y1),
                                  transform=fig.transFigure,
                                  arrowstyle="->", mutation_scale=14,
                                  color="#444", linewidth=1.2)
    )
    if label:
        fig.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.02, label,
                 ha="center", fontsize=8, color="#444", style="italic")


# ---------------------------------------------------------------------------
# 1. Brutal
# ---------------------------------------------------------------------------
def make_brutal():
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    fig.suptitle("Brutal — single-shot summarization", fontsize=12, fontweight="bold")

    # BEFORE: full context, near 90%
    before = [
        (0,   178, C_RAW,    "Raw messages (~178K tokens)"),
        (178, 180, C_RECENT, "Last 2 msgs"),
    ]
    draw_panel(axes[0], "BEFORE (ctx ≥ 90%)", before,
               watermarks=[(180, C_HIGH_WM, "HIGH WM (90%)"),
                           (120, C_LOW_WM,  "LOW WM (60%)")])

    # AFTER: empty + last 2 msgs + summary
    after = [
        (0,   3,   C_SUMMARY, "Summary (N) — ~500 tok", True),
        (3,   5,   C_RECENT,  "Last 2 msgs", True),
        (5,   180, C_EMPTY,   "", False),
    ]
    draw_panel(axes[1], "AFTER", after,
               watermarks=[(180, C_HIGH_WM, "HIGH WM (90%)"),
                           (120, C_LOW_WM,  "LOW WM (60%)")])

    fig.text(0.5, 0.55, "summarize ALL\n(truncated at ~150K cap)",
             ha="center", va="center", fontsize=8, style="italic",
             color="#444",
             bbox=dict(facecolor="white", edgecolor="#888",
                       boxstyle="round,pad=0.3"))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "fig_strategy_brutal.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Incremental
# ---------------------------------------------------------------------------
def make_incremental():
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    fig.suptitle("Incremental — re-summarizes prior summary (JPEG cascade)",
                 fontsize=12, fontweight="bold")

    before = [
        (0,   2,   C_SUMMARY, "Summary v.(N−1) ~2K"),
        (2,   180, C_RAW,     "Raw messages"),
    ]
    draw_panel(axes[0], "BEFORE (ctx ≥ 90%)", before,
               watermarks=[(180, C_HIGH_WM, "HIGH WM (90%)"),
                           (120, C_LOW_WM,  "LOW WM (60%)")])

    after = [
        (0,   2,   C_SUMMARY, "Summary v.N (re-summarized)", True),
        (2,   120, C_RECENT,  "Recent messages kept (~118K)", False),
        (120, 180, C_EMPTY,   "", False),
    ]
    draw_panel(axes[1], "AFTER (compact down to 60%)", after,
               watermarks=[(180, C_HIGH_WM, "HIGH WM (90%)"),
                           (120, C_LOW_WM,  "LOW WM (60%)")])

    fig.text(0.5, 0.55, "summarize old +\nprior summary together",
             ha="center", va="center", fontsize=8, style="italic",
             color="#444",
             bbox=dict(facecolor="white", edgecolor="#888",
                       boxstyle="round,pad=0.3"))

    fig.text(0.5, 0.02,
             "Problem: each cycle, summary v.N is a summary of a summary "
             "of a summary…",
             ha="center", fontsize=8.5, style="italic", color="#b71c1c")

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(OUT / "fig_strategy_incremental.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Frozen
# ---------------------------------------------------------------------------
def make_frozen():
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    fig.suptitle("Frozen — immutable summaries, raw messages compacted",
                 fontsize=12, fontweight="bold")

    before = [
        (0,   12,  C_FROZEN, "Frozen #1, #2"),
        (12,  60,  C_FROZEN, "Frozen #(N−1)"),
        (60,  180, C_RAW,    "Raw messages"),
    ]
    draw_panel(axes[0], "BEFORE (ctx ≥ 90%)", before,
               watermarks=[(180, C_HIGH_WM, "HIGH WM (90%)"),
                           (120, C_LOW_WM,  "LOW WM (60%)")],
               budget=(60, "FROZEN BUDGET (30%)"))

    after = [
        (0,   12,  C_FROZEN, "Frozen #1, #2"),
        (12,  60,  C_FROZEN, "Frozen #(N−1)"),
        (60,  72,  C_FROZEN, "NEW Frozen #N", True),
        (72,  120, C_RECENT, "Recent kept (~74K)"),
        (120, 180, C_EMPTY,  ""),
    ]
    draw_panel(axes[1], "AFTER", after,
               watermarks=[(180, C_HIGH_WM, "HIGH WM (90%)"),
                           (120, C_LOW_WM,  "LOW WM (60%)")],
               budget=(60, "FROZEN BUDGET (30%)"))

    fig.text(0.5, 0.02,
             "Frozen summaries are never re-summarized. Tradeoff: they "
             "consume context space.",
             ha="center", fontsize=8.5, style="italic", color="#1b5e20")

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(OUT / "fig_strategy_frozen.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. FrozenRanked
# ---------------------------------------------------------------------------
def make_frozenranked():
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.2))
    fig.suptitle("FrozenRanked — hierarchical merge (tournament bracket)",
                 fontsize=12, fontweight="bold")

    # Left: Frozen sequential merge
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Frozen — sequential\n(merge oldest pair)",
                 fontsize=10, fontweight="bold")
    labels_left = [
        ("#6", "newest", C_FROZEN),
        ("#5", "",       C_FROZEN),
        ("#4", "",       C_FROZEN),
        ("#3", "",       C_FROZEN),
        ("#2", "",       C_FROZEN),
        ("#1", "oldest", C_FROZEN),
    ]
    for i, (n, tag, c) in enumerate(labels_left):
        y = 5.5 - i
        ax.add_patch(mpatches.Rectangle((0.20, y), 0.50, 0.85,
                                        facecolor=c, edgecolor="black"))
        ax.text(0.45, y + 0.42, f"Frozen {n}", ha="center", va="center",
                fontsize=9)
        if tag:
            ax.text(0.74, y + 0.42, f"◄ {tag}", ha="left", va="center",
                    fontsize=8, color="#555")
    # arrow "merge oldest pair"
    ax.annotate("", xy=(0.18, 0.4), xytext=(0.18, 1.4),
                arrowprops=dict(arrowstyle="->", color=C_HIGH_WM, lw=1.8))
    ax.text(0.05, 0.9, "merge\n#1+#2", fontsize=8, color=C_HIGH_WM,
            fontweight="bold")
    ax.text(0.5, -0.1,
            "After 10 merges: #1 has been re-summarized\n5× (≈ N/2) — severe degradation",
            ha="center", fontsize=8, color="#b71c1c", style="italic")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Right: FrozenRanked
    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("FrozenRanked — by rank\n(merge two lowest-rank)",
                 fontsize=10, fontweight="bold")
    labels_right = [
        ("#6", "R1", "newest", C_FROZEN),
        ("#5", "R1", "",       C_FROZEN),
        ("#4", "R1", "",       C_FROZEN),
        ("#3", "R2", "(R1+R1)", C_FROZEN_R2),
        ("#2", "R2", "(R1+R1)", C_FROZEN_R2),
        ("#1", "R3", "(R2+R2)", C_FROZEN_R3),
    ]
    for i, (n, rank, tag, c) in enumerate(labels_right):
        y = 5.5 - i
        ax.add_patch(mpatches.Rectangle((0.15, y), 0.40, 0.85,
                                        facecolor=c, edgecolor="black"))
        text_color = "white" if c == C_FROZEN_R3 else "black"
        ax.text(0.35, y + 0.42, f"Frozen {n}", ha="center", va="center",
                fontsize=9, color=text_color)
        ax.add_patch(mpatches.Rectangle((0.56, y), 0.10, 0.85,
                                        facecolor="#eeeeee", edgecolor="black"))
        ax.text(0.61, y + 0.42, rank, ha="center", va="center", fontsize=8,
                fontweight="bold")
        if tag:
            ax.text(0.69, y + 0.42, tag, ha="left", va="center", fontsize=7.5,
                    color="#555")
    # arrow "merge two lowest-rank"
    ax.annotate("", xy=(0.13, 4.4), xytext=(0.13, 5.4),
                arrowprops=dict(arrowstyle="->", color="#1b5e20", lw=1.8))
    ax.text(0.005, 4.9, "merge\n#5+#6\n(both R1)", fontsize=8,
            color="#1b5e20", fontweight="bold")
    ax.text(0.5, -0.1,
            "After 10 merges: #1 re-summarized\n3× (≈ log₂N) — moderate degradation",
            ha="center", fontsize=8, color="#1b5e20", style="italic")
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout(rect=[0, 0.06, 1, 0.93])
    fig.savefig(OUT / "fig_strategy_frozenranked.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_brutal()
    make_incremental()
    make_frozen()
    make_frozenranked()
    print("Saved 4 strategy diagrams to:", OUT)
