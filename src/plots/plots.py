"""The chart functions the analyses call."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, FuncNorm
from matplotlib.ticker import FuncFormatter

from .theme import (ACCENT, AXIS, CLASS_NEG, CLASS_POS, DUMBBELL_RIGHT, GAINED, GRIDLINE,
                    INK_MUTED, INK_PRIMARY, INK_SECONDARY, LOST, REPRESENTATION_COLOUR,
                    SEQUENTIAL_BLUE, SERIES, SURFACE, UNCHANGED, _place_header, _shorten,
                    _wrap_to_width)


# ----------------- CREATE FEATURE IMPORTANCE RANK SHIFT PLOTS ------------------------------------------

RANK_KNEE = 50.0          # ranks up to here keep their true spacing on the y-axis
RANK_TAIL_FACTOR = 0.15   # ranks past the knee are compressed by this factor
SLOPE_LABEL_FONTSIZE = 8  # the direct end labels; also sets their minimum spacing


def _spread_labels(y_positions: np.ndarray, min_gap: float) -> np.ndarray:
    """
    Given the markers y position and required minimum spacing, calculate the spreaded y-positions of the labels where no two labels are closer thatn min_gap. Vertical order is kept.

    """
    order = np.argsort(y_positions, kind="stable")  # top (small y) to bottom
    y = np.asarray(y_positions, dtype=float)[order]
    # push the overlapping labels down. Go through labels from top to bottom. Label may be pushed off the bottom.
    for i in range(1, len(y)):
        if y[i] - y[i - 1] < min_gap:
            y[i] = y[i - 1] + min_gap
    # go through labels from bottom to top. If label to close to the one below, push up. Kind of recentres the cluster by pulling the labels up again.
    for i in range(len(y) - 2, -1, -1):
        if y[i + 1] - y[i] < min_gap:
            y[i] = y[i + 1] - min_gap
    spread = np.empty_like(y)
    spread[order] = y # undo the sorting
    return spread


def _compress_rank(rank):
    """
    Up to rank 50 it returns identity. Below rank 50 (RANK_KNEE) compresses the positioning by RANK-TAIL_FACTOR.
    """
    rank = np.asarray(rank, dtype=float)
    compressed = np.where(rank <= RANK_KNEE, rank,
                          RANK_KNEE + (rank - RANK_KNEE) * RANK_TAIL_FACTOR)
    return compressed if compressed.ndim else compressed.item() # handle the case if function is called with scalar (rank=50 for the kneeline). Then np.where returns 0d array


def _slope_panel(ax, panel_df: pd.DataFrame, left_label: str, right_label: str,
                 ylim: tuple[float, float]) -> None:
    """
    Draw one slope chart: a line per row of panel_df, from its rank under the left
    representation to its rank under the right one.
    :param panel_df: columns label, rank_left, rank_right
    :param ylim: shared across panels, so equal slopes mean equal rank shifts
    """
    occupied_ranks = pd.unique(pd.concat([panel_df["rank_left"], panel_df["rank_right"]])) # ranks that are actually exist in the data
    ax.hlines(_compress_rank(occupied_ranks), xmin=0, xmax=1, color=GRIDLINE,
              linewidth=0.8, zorder=1)
    for x in (0, 1):
        ax.axvline(x, color=GRIDLINE, linewidth=0.8, zorder=1)
    # mark where the tail starts being compressed, if the panel reaches into it. Only make this dashed line if its reached past rank 50
    if ylim[0] > RANK_KNEE:
        ax.axhline(RANK_KNEE, color=AXIS, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate("rank 50 · scale compressed below", (-0.62, RANK_KNEE),
                    xytext=(0, 3), textcoords="offset points", ha="left", va="bottom",
                    fontsize=7, color=INK_MUTED)
    # get marker heights and compressed marker heights
    labels = panel_df["label"].to_list()
    rank_left = panel_df["rank_left"].to_numpy()
    rank_right = panel_df["rank_right"].to_numpy()
    y_left = _compress_rank(rank_left)   # true (compressed) marker heights
    y_right = _compress_rank(rank_right)
    # flag unused labs as "unused" instead of giving them a rank
    n = len(labels)
    unused_left = (panel_df["unused_left"].to_numpy() if "unused_left" in panel_df
                   else np.zeros(n, dtype=bool))
    unused_right = (panel_df["unused_right"].to_numpy() if "unused_right" in panel_df
                    else np.zeros(n, dtype=bool))

    # draw the slopes and colourise depending on direction
    for label, rl, rr, yl, yr in zip(labels, rank_left, rank_right, y_left, y_right):
        if rr == rl:
            colour = UNCHANGED
        else:
            colour = GAINED if rr < rl else LOST  # smaller rank = more important
        ax.plot([0, 1], [yl, yr], color=colour, linewidth=1.6,
                marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2,
                zorder=3)  # surface ring keeps overlapping markers separated

    # decouple labels from markers to avoid label collision.
    axis_height_pt = ax.get_position().height * ax.figure.get_figheight() * 72.0 # get real height of the axis in points
    data_range = abs(ylim[0] - ylim[1])
    min_gap = SLOPE_LABEL_FONTSIZE * 1.4 * data_range / max(axis_height_pt, 1.0) # define min gap between the labels
    min_gap = min(min_gap, 0.92 * data_range / max(len(labels), 1))  # never overflow the axis
    # spread the labels (min_gap between the labels) -> get new label positions
    label_left = _spread_labels(y_left, min_gap)
    label_right = _spread_labels(y_right, min_gap)
    # draw the labels and the connecting lines
    for label, rl, rr, yl, yr, ll, lr, ul, ur in zip(labels, rank_left, rank_right,
                                                     y_left, y_right, label_left,
                                                     label_right, unused_left, unused_right):
        left_rank = "unused" if ul else f"{rl:.0f}"
        right_rank = "unused" if ur else f"{rr:.0f}"
        ax.plot([-0.1, 0], [ll, yl], color=GRIDLINE, linewidth=0.6, zorder=2)
        ax.plot([1, 1.1], [yr, lr], color=GRIDLINE, linewidth=0.6, zorder=2)
        ax.annotate(f"{label}  {left_rank}", (-0.1, ll), xytext=(-2, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=SLOPE_LABEL_FONTSIZE, color=INK_SECONDARY, annotation_clip=False)
        ax.annotate(f"{right_rank}  {label}", (1.1, lr), xytext=(2, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=SLOPE_LABEL_FONTSIZE, color=INK_SECONDARY, annotation_clip=False)

    ax.set_ylim(*ylim)  # rank 1 at the top
    ax.set_xlim(-0.72, 1.72)

    # the two labelled columns are the axis
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([left_label, right_label], fontsize=10, color=INK_PRIMARY)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_yticks([])


def plot_rank_shift_slope_chart(panels: list[tuple[str, str, pd.DataFrame]],
                                title: str, subtitle: str, path) -> None:
    """
    One figure with a slope chart per panel, side by side on a shared rank axis.
    :param panels: (left_label, right_label, panel_df) triples, one per comparison.
                   panel_df has one row per feature: label, rank_left, rank_right.
    :param title: figure heading
    :param subtitle: one line under the heading, e.g. what was selected
    :param path: pathlib.Path the png is written to
    """
    # computes shared y scale across all panels
    all_ranks = pd.concat([panel_df[["rank_left", "rank_right"]] for _, _, panel_df in panels])
    lo, hi = _compress_rank(all_ranks.min().min()), _compress_rank(all_ranks.max().max())
    pad = 0.04 * (hi - lo)
    ylim = (hi + pad, lo - pad)

    # size the figure according to the panel with the most items (at most 40 items)
    max_rows = max(len(panel_df) for _, _, panel_df in panels)
    height = min(13.0, max(8.4, 0.28 * max_rows + 4.0))
    fig, axes = plt.subplots(1, len(panels), figsize=(7.3 * len(panels), height),
                             facecolor=SURFACE, gridspec_kw={"wspace": 0.75})
    # create the individual panels
    for ax, (left_label, right_label, panel_df) in zip(np.atleast_1d(axes), panels):
        _slope_panel(ax, panel_df, left_label, right_label, ylim)

    # overall figure legend and labels and headers...
    handles = [plt.Line2D([], [], color=GAINED, linewidth=1.6, label="gained importance"),
               plt.Line2D([], [], color=LOST, linewidth=1.6, label="lost importance"),
               plt.Line2D([], [], color=UNCHANGED, linewidth=1.6, label="unchanged")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9, labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, -0.02))
    _place_header(fig, title, subtitle)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved rank shift slope chart to {path}.")

# -----------DUMBBELL PLOTS ------------------------------------------------------------
# helper for creation of a singledumbbell panel
def _dumbbell_panel(ax, y: np.ndarray, left: np.ndarray, right: np.ndarray,
                    left_colour: str, right_colour: str) -> None:
    """
    Draw one dumbbell panel into ax
    """
    n_rows = len(y)
    for yi, l, r in zip(y, left, right):
        if np.isnan(l) or np.isnan(r):
            continue
        ax.plot([l, r], [yi, yi], color=AXIS, linewidth=2.2, zorder=2,
                solid_capstyle="round")
    # NaNs are not plotted
    ax.scatter(left, y, s=80, color=left_colour, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.scatter(right, y, s=80, color=right_colour, edgecolor=SURFACE, linewidth=1.5, zorder=3)

    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_yticks(y)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    # hairline between each row
    for boundary in range(1, n_rows):
        ax.axhline(boundary - 0.5, color=AXIS, linewidth=0.8, zorder=1)

# dumbbell chart for per-cohort equalised odds
def plot_dumbbell_chart(panel_df: pd.DataFrame, left_col: str, right_col: str,
                        left_label: str, right_label: str, title: str, subtitle: str,
                        xlabel: str, path, note: str | None = None) -> None:
    """
    :param panel_df: indexed by row label, with numeric columns left_col and right_col.
    :param xlabel: what the shared x-axis measures.
    :param path: pathlib.Path the png is written to.
    :param note: optional one-line footnote under the figure
    """
    rows = panel_df.index.tolist()
    y = np.arange(len(rows))[::-1]  # first row at the top
    left = panel_df[left_col].to_numpy(dtype=float)
    right = panel_df[right_col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 0.55 * len(rows) + 2.0), facecolor=SURFACE)
    _dumbbell_panel(ax, y, left, right, ACCENT, DUMBBELL_RIGHT)

    ax.set_yticklabels(rows, fontsize=9, color=INK_SECONDARY)
    ax.set_xlim(left=0.0)  # a gap is a distance from 0
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)

    handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=9,
                          markerfacecolor=ACCENT, markeredgecolor=SURFACE, label=left_label),
               plt.Line2D([], [], marker="o", linestyle="none", markersize=9,
                          markerfacecolor=DUMBBELL_RIGHT, markeredgecolor=SURFACE, label=right_label)]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=9, labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, -0.02))
    _place_header(fig, title, subtitle)
    if note:
        fig.text(0.01, -0.06, note, color=INK_MUTED, fontsize=8, ha="left")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved dumbbell chart to {path}.")

# dumbbell plot for equalised odds across cohorts.
def plot_fairness_eo_dumbbell(gaps: pd.DataFrame, baseline: tuple[str, str],
                              targets: list[tuple[str, str]], title: str, subtitle: str,
                              xlabel: str, path, note: str | None = None) -> None:
    # get rows
    cohorts = list(gaps.index)
    y = np.arange(len(cohorts))[::-1]  # first cohort at the top
    base_key, base_label = baseline
    # get shared x-axis
    finite = gaps.to_numpy(dtype=float)
    xmax = np.nanmax(finite) if np.isfinite(np.nanmax(finite)) else 1.0
    xmax = xmax * 1.08 if xmax > 0 else 1.0

    # make one panel per target (represenation comparison)
    fig, axes = plt.subplots(1, len(targets), sharey=True, facecolor=SURFACE,
                             figsize=(5.0 * len(targets) + 1.0, 0.55 * len(cohorts) + 2.6),
                             gridspec_kw={"wspace": 0.1})
    for i, (ax, (tgt_key, tgt_label)) in enumerate(zip(np.atleast_1d(axes), targets)):
        left = gaps[base_key].to_numpy(dtype=float)
        right = gaps[tgt_key].to_numpy(dtype=float)
        _dumbbell_panel(ax, y, left, right,
                        REPRESENTATION_COLOUR[base_key], REPRESENTATION_COLOUR[tgt_key])
        ax.set_xlim(0.0, xmax)  # a gap is a distance from 0; shared across panels
        if i == 0:
            ax.set_yticklabels(cohorts, fontsize=9, color=INK_SECONDARY)
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
        ax.set_title(f"{base_label} vs {tgt_label}", color=INK_PRIMARY, fontsize=11, pad=10)

    handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=9,
                          markerfacecolor=REPRESENTATION_COLOUR[key], markeredgecolor=SURFACE,
                          label=label)
               for key, label in [baseline, *targets]]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=9, labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, -0.08))
    _place_header(fig, title, subtitle)
    if note:
        # below the legend (which sits at -0.08), so the two do not overlap
        fig.text(0.01, -0.13, note, color=INK_MUTED, fontsize=8, ha="left")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved fairness equalised-odds dumbbell to {path}.")

# ---------------- PER-SAMPLE CLASSIFICATION HEATMAP --------------------------------------------------

def plot_classification_heatmap(data: np.ndarray, row_labels: list[str],
                                n_ground_truth_rows: int, split_x: float | None,
                                title: str, subtitle: str, xlabel: str, path) -> None:
    """
    Heatmap of binary outcomes: one row per label in row_labels, one column per
    sample, cells coloured by class (0 = CLASS_NEG, 1 = CLASS_POS).
    """
    # colour map
    n_rows, n_cols = data.shape
    cmap = ListedColormap([CLASS_NEG, CLASS_POS])
    cmap.set_bad(SURFACE)
    # draw grid
    fig, ax = plt.subplots(figsize=(12.0, 0.42 * n_rows + 1.8), facecolor=SURFACE)
    ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=0, vmax=1,
              aspect="auto", interpolation="nearest")

    # row separators
    for boundary in range(1, n_rows):
        ax.axhline(boundary - 0.5, color=SURFACE, linewidth=1.5, zorder=3)
        ax.axhline(boundary - 0.5, color=GRIDLINE, linewidth=0.6, zorder=4)
    # darker line below ground truth
    if 0 < n_ground_truth_rows < n_rows:
        ax.axhline(n_ground_truth_rows - 0.5, color=SURFACE, linewidth=4, zorder=5)
        ax.axhline(n_ground_truth_rows - 0.5, color=INK_PRIMARY, linewidth=1.0, zorder=6)
    if split_x is not None:
        ax.axvline(split_x, color=INK_PRIMARY, linewidth=1.0, alpha=0.55, zorder=4)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9, color=INK_SECONDARY)
    ax.set_xticks([])
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                          markerfacecolor=CLASS_NEG, markeredgecolor=GRIDLINE,
                          label="label 0 (negative)"),
               plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                          markerfacecolor=CLASS_POS, markeredgecolor=GRIDLINE,
                          label="label 1 (positive)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    _place_header(fig, title, subtitle)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved classification heatmap to {path}.")

# ------------- BUMP CHART FOR FEATURE IMPORTANCE RANK ACROSS FOLDS -----------------------------------

BUMP_GROUP_SIZE = 5  # labs are coloured in importance blocks of this size
def _bump_group_colours(panel_df: pd.DataFrame) -> dict:
    """
    Map each feature label to a colour based on its importance group.
    """
    ordered = panel_df.mean(axis=1).sort_values(kind="stable").index
    return {label: SERIES[(i // BUMP_GROUP_SIZE) % len(SERIES)]
            for i, label in enumerate(ordered)}

def _bump_panel(ax, panel_df: pd.DataFrame, panel_title: str, highlighted: list,
                ylim: tuple[float, float]) -> None:
    """
    Draw one bump chart
    :param panel_df: index = feature label, columns = folds, values = rank (1 = most important)
    :param highlighted: labels to name directly at the end of their line.
    """
    folds = panel_df.columns.to_numpy()
    colours = _bump_group_colours(panel_df) # get colour per labitem
    # draw
    for label in reversed(panel_df.mean(axis=1).sort_values(kind="stable").index): # iterate in reversed order to plot most important labs last. So they arent hidden by least important labs.
        ranks = panel_df.loc[label].to_numpy()
        ax.plot(folds, ranks, color=colours[label], linewidth=1.6, marker="o",
                markersize=4, markeredgecolor=SURFACE, markeredgewidth=1.5,
                alpha=0.9, zorder=2)
    # name all items
    highlighted = set(highlighted) # top 5 items by average
    for label in panel_df.index:
        ranks = panel_df.loc[label].to_numpy()
        ax.annotate(f" {_shorten(str(label))}", (folds[-1], ranks[-1]),
                    xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=8, color=INK_SECONDARY,
                    fontweight="bold" if label in highlighted else "normal",
                    annotation_clip=False)

    ax.set_ylim(*ylim)  # rank 1 at the top
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRIDLINE)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(folds)
    ax.set_xlabel("fold", color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.set_title(panel_title, color=INK_PRIMARY, fontsize=11, pad=10)


def plot_rank_bump_chart(panels: list[tuple[str, pd.DataFrame, list]], title: str,
                         subtitle: str, path) -> None:
    """
    One figure with a bump chart per panel, side by side on a shared rank axis:
    how each feature's importance rank moves across the folds.
    :param panels: (panel_title, panel_df, highlighted) triples. panel_df is indexed
                   by feature label, one column per fold, values are ranks.
                   highlighted are that panel's own labels to accent and name.
    :param title: figure heading
    :param subtitle: one line under the heading, e.g. what was selected
    :param path: pathlib.Path the png is written to
    """
    # get shared rank scale across all panels
    all_ranks = pd.concat([panel_df for _, panel_df, _ in panels])
    ylim = (all_ranks.to_numpy().max() + 0.5, 0.5)

    # create subplots and call helper
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 7.6),
                             sharey=True, facecolor=SURFACE,
                             gridspec_kw={"wspace": 0.85})
    for ax, (panel_title, panel_df, highlighted) in zip(np.atleast_1d(axes), panels):
        _bump_panel(ax, panel_df, panel_title, highlighted, ylim)

    # Legends, title etc.
    first = np.atleast_1d(axes)[0]
    first.set_ylabel("Importance rank (1 = most important)", color=INK_SECONDARY, fontsize=9)

    # one legend entry per importance block
    n_features = max(len(panel_df) for _, panel_df, _ in panels)
    n_groups = min(-(-n_features // BUMP_GROUP_SIZE), len(SERIES))
    handles = [plt.Line2D([], [], color=SERIES[g], linewidth=1.6,
                          label=f"Importance (averaged across folds) {g * BUMP_GROUP_SIZE + 1}"
                                f"–{(g + 1) * BUMP_GROUP_SIZE}")
               for g in range(n_groups)]
    fig.legend(handles=handles, loc="lower center", ncol=n_groups, frameon=False,
               fontsize=9, labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, -0.04))

    _place_header(fig, title, subtitle)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved rank bump chart to {path}.")


def plot_value_heatmap(df: pd.DataFrame, title: str, subtitle: str,
                       colourbar_label: str, path,
                       separate_last_row: bool = False,
                       diverging: bool = False) -> None:
    """
    Heatmap of a value (Kendalls W or performance measures)
    :param diverging: for signed values such as a difference to a baseline. Uses the
                      red/white/blue ramp on a symmetric range, so 0 is white and the two
                      directions are equally visible. Otherwise the sequential blue ramp
                      spans the observed range.
    """
    values = df.to_numpy(dtype=float)
    if diverging:
        # symmetric around 0, so a positive and a negative of equal size read equally strong
        limit = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
        if not np.isfinite(limit) or limit == 0:
            limit = 1.0  # degenerate all-zero/empty case: keep a valid range
        vmin, vmax = -limit, limit
        cmap = LinearSegmentedColormap.from_list("thresh_diverging", [LOST, "#ffffff", GAINED])
    else:
        # colour scale spans observed range to make small differences visible
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        cmap = LinearSegmentedColormap.from_list("thresh_blue", SEQUENTIAL_BLUE)
    cmap.set_bad(SURFACE)

    # a blank spacer row before the last row to set average row apart from cohort rows
    row_labels = list(df.index)
    if separate_last_row and len(values) > 1:
        values = np.insert(values, len(values) - 1, np.full(values.shape[1], np.nan), axis=0)
        row_labels.insert(len(row_labels) - 1, "")

    # draw the grid and colourise the cells
    n_rows, n_cols = values.shape
    fig, ax = plt.subplots(figsize=(1.9 * n_cols + 3.4, 0.72 * n_rows + 2.2),
                           facecolor=SURFACE)
    ax.pcolormesh(np.ma.masked_invalid(values), cmap=cmap, vmin=vmin, vmax=vmax,
                  edgecolors=SURFACE, linewidth=2)  # surface gap between cells

    # print value in each cell
    midpoint = (vmin + vmax) / 2
    for row in range(n_rows):
        for col in range(n_cols):
            value = values[row, col]
            if np.isnan(value):
                continue  # spacer row is skipped
            # text colour flips depending on cell colour. On the diverging ramp the dark
            # cells are at BOTH ends, so the flip goes by distance from the white centre
            dark = abs(value) > 0.55 * vmax if diverging else value > midpoint
            ax.annotate(f"{value:+.3f}" if diverging else f"{value:.3f}",
                        (col + 0.5, row + 0.5), ha="center", va="center",
                        fontsize=9, color=SURFACE if dark else INK_PRIMARY)
    # labels etc
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_xticklabels(df.columns, fontsize=9, color=INK_SECONDARY)
    ax.set_yticklabels(row_labels, fontsize=9, color=INK_SECONDARY)
    ax.invert_yaxis()  # first cohort at the top
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colourbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap,
                                                   norm=plt.Normalize(vmin, vmax)),
                             ax=ax, fraction=0.03, pad=0.03)
    colourbar.set_label(colourbar_label, color=INK_SECONDARY, fontsize=9)
    colourbar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    colourbar.outline.set_visible(False)

    _place_header(fig, title, subtitle)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved value heatmap to {path}.")


def plot_representation_boxplot(values_df: pd.DataFrame, groups, group_order: list,
                                series: list, title: str, subtitle: str, ylabel: str, path,
                                hline: tuple | None = None,
                                colours: dict[str, str] | None = None,
                                reference_series: str | None = None) -> None:
    """
    One box per representation, repeated for every group, with every cohort drawn as a dot.
    The box spans the interquartile range, the line in it is the median and the whiskers
    reach the minimum and maximum, so no value is cut off or marked an outlier.
    :param values_df: rows = cohort, columns = representation key
    :param groups: Series [cohort -> group key], or None for a single ungrouped set of boxes
    :param group_order: group keys in plot order, ignored when groups is None
    :param series: list of (representation key, display label)
    :param hline: (y, label) for a horizontal reference rule
    :param colours: series key -> colour, for keys REPRESENTATION_COLOUR does not hold
        (a per-strategy variant, say). Defaults to the project-wide representation colours.
    :param reference_series: series key whose quartiles are extended across the whole axis
        as rules, so every other box can be read against that spread by eye
    """
    colour_of = {**REPRESENTATION_COLOUR, **(colours or {})}
    n_reps = len(series)
    ungrouped = groups is None
    keys = ["all"] if ungrouped else [k for k in group_order if (groups == k).any()]
    if ungrouped:
        groups = pd.Series("all", index=values_df.index)

    if ungrouped:
        longest = max((len(label) for _, label in series), default=8)
        per_box = max(0.62, longest * 0.075)
    else:
        per_box = 0.26
    box_width = 0.72 if not ungrouped else min(0.72, 0.62 / per_box)  # keep boxes slim
    fig_width = max(6.5, 2.4 + len(keys) * (n_reps * per_box + max(0.5, per_box)))
    # a grouped figure is wide by nature, so give it more height as well, or the axis
    # labels end up tiny against a very long axis
    fig_height = 5.2 if ungrouped else min(7.0, 5.2 + 0.08 * len(keys))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=SURFACE)
    rng = np.random.default_rng(0)  # deterministic jitter, so reruns are identical

    tick_positions, tick_labels = [], []
    for group_index, key in enumerate(keys):
        members = groups.index[groups == key]
        centre = group_index * (n_reps + 1.0)
        for rep_index, (rep, label) in enumerate(series):
            position = centre + rep_index
            values = values_df.loc[members, rep].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue  # a representation with no data yet would otherwise raise
            colour = colour_of[rep]
            ax.boxplot([values], positions=[position], widths=box_width,
                       whis=(0, 100), showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=colour, edgecolor=colour, alpha=0.22,
                                     linewidth=1.0),
                       medianprops=dict(color=colour, linewidth=2.0),
                       whiskerprops=dict(color=colour, linewidth=1.0),
                       capprops=dict(color=colour, linewidth=1.0), zorder=2)
            # jitter the dots so equal values do not hide behind one another
            jitter = rng.uniform(-0.18, 0.18, len(values)) if len(values) > 1 else np.zeros(1)
            ax.scatter(position + jitter, values, s=18, color=colour, alpha=0.85,
                       edgecolors=SURFACE, linewidths=0.6, zorder=3)
            if ungrouped:
                tick_positions.append(position)
                tick_labels.append(label)
        if not ungrouped:
            tick_positions.append(centre + (n_reps - 1) / 2)
            tick_labels.append(f"{key}\n(n={len(members)})")
            # a rule in the gap after this group, so the eye reads one group as one block
            # rather than counting boxes. The gap is one slot wide, so its middle sits a
            # full n_reps from the group's first box. None after the last group.
            if group_index < len(keys) - 1:
                ax.axvline(centre + n_reps, color=AXIS, linewidth=0.8, zorder=1)

    # the reference representation's own box extended across the plot. Same quartile
    # definition matplotlib's boxplot uses (linear interpolation), so the rules line up
    # exactly with the edges and median of that representation's box
    if reference_series is not None and reference_series in values_df.columns:
        reference_values = values_df[reference_series].dropna().to_numpy(dtype=float)
        if len(reference_values):
            reference_colour = colour_of[reference_series]
            q1, median, q3 = np.percentile(reference_values, [25, 50, 75])
            reference_label = dict(series).get(reference_series, reference_series)
            # the median gets the heavier rule, the quartiles a lighter one. The labels go
            # just outside the right spine -- inside they would be read against whichever
            # box happens to sit at that end -- and are nudged apart vertically so a narrow
            # interquartile range does not stack them on top of one another
            for y_value, dashes, line_width, text, valign in [
                    (q3, (0, (4, 3)), 1.0, "Q3", "bottom"),
                    (median, (0, (6, 2)), 1.6, f"{reference_label} median", "center"),
                    (q1, (0, (4, 3)), 1.0, "Q1", "top")]:
                ax.axhline(y_value, color=reference_colour, linewidth=line_width,
                           linestyle=dashes, alpha=0.7, zorder=1)
                ax.annotate(text, (1.0, y_value), xycoords=("axes fraction", "data"),
                            xytext=(5, {"bottom": 2, "center": 0, "top": -2}[valign]),
                            textcoords="offset points", ha="left", va=valign,
                            fontsize=8, color=reference_colour,
                            annotation_clip=False)  # outside the axes, so do not clip

    if hline is not None:
        y_value, h_label = hline
        ax.axhline(y_value, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        # an empty label draws the rule bare, for when the axis label already says what
        # zero means and the annotation would only repeat it
        if h_label:
            # at the left edge: the boxes start a little in from it, so nothing is overwritten
            ax.annotate(h_label, (0.005, y_value), xycoords=("axes fraction", "data"),
                        xytext=(0, 4), textcoords="offset points", ha="left",
                        fontsize=8, color=INK_MUTED)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
        spine.set_linewidth(0.8)

    # a single series needs no legend: the title says what the box is. With several, the
    # x axis labels the groups and the legend has to carry the series identity.
    if not ungrouped and len(series) > 1:
        handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                              markerfacecolor=colour_of[rep],
                              markeredgecolor=SURFACE, label=label)
                   for rep, label in series]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                   fontsize=9, labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, -0.06))
    _place_header(fig, title, subtitle)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved representation boxplot to {path}.")


def plot_performance_scatter(perf_df: pd.DataFrame, panels: list, series: list,
                             title: str, subtitle: str, ylabel: str, path,
                             hline: tuple | None = None,
                             property_panel: tuple | None = None) -> None:
    """
    One dot per cohort per representation against a property of the cohort, with a least
    squares fit per representation. Keeps the per-cohort spread that a binned heatmap
    averages away, so a gradient can be told apart from a threshold.
    :param perf_df: rows = cohort, columns = representation key, values = the measure
    :param panels: list of (pd.Series [cohort -> x], x axis label, log_x)
    :param series: list of (representation key, display label)
    :param hline: (y, label) for a horizontal reference rule, e.g. an inclusion threshold
    :param property_panel: optional (x series, y series, x label, y label, log_x) plotting
        two cohort properties against each other. It measures neither the performance nor
        the representations, so it gets its own y axis, one neutral dot per cohort and no
        series: what it is for is showing whether the x axes of the other panels are
        themselves related, which decides whether their slopes can be read separately.
    """
    n_panels = len(panels) + (1 if property_panel is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.9 * n_panels, 4.9), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    measure_axes = axes[:len(panels)]
    # the performance panels share one y axis; the property panel deliberately does not,
    # so give them the shared scale by hand rather than with sharey
    for ax in measure_axes[1:]:
        ax.sharey(measure_axes[0])
        ax.tick_params(labelleft=False)

    for ax, (x_series, x_label, log_x) in zip(measure_axes, panels):
        x = x_series.reindex(perf_df.index).to_numpy(dtype=float)
        # fit in the plotted space, so the line matches what the eye sees on a log axis
        fit_x = np.log10(x) if log_x else x
        for key, _ in series:
            y = perf_df[key].to_numpy(dtype=float)
            colour = REPRESENTATION_COLOUR[key]
            ax.scatter(x, y, s=26, color=colour, alpha=0.75, edgecolors=SURFACE,
                       linewidths=0.8, zorder=3)
            ok = np.isfinite(fit_x) & np.isfinite(y)
            if ok.sum() >= 3:  # two points would draw a fit through nothing
                slope, intercept = np.polyfit(fit_x[ok], y[ok], 1)
                grid = np.linspace(fit_x[ok].min(), fit_x[ok].max(), 100)
                ax.plot(10 ** grid if log_x else grid, slope * grid + intercept,
                        color=colour, linewidth=1.4, alpha=0.9, zorder=2)
        if log_x:
            ax.set_xscale("log")
        if hline is not None:
            y_value, h_label = hline
            ax.axhline(y_value, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
            # annotate once, on the rightmost panel, where the fits have left room, so the
            # rule is neither labelled twice nor written over a line
            if ax is measure_axes[-1]:
                ax.annotate(h_label, (0.99, y_value), xycoords=("axes fraction", "data"),
                            xytext=(0, 4), textcoords="offset points", ha="right",
                            fontsize=8, color=INK_MUTED)
        ax.set_xlabel(x_label, color=INK_SECONDARY, fontsize=9)
        ax.grid(axis="both", color=GRIDLINE, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
        for spine in ax.spines.values():
            spine.set_color(AXIS)
            spine.set_linewidth(0.8)

    measure_axes[0].set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)

    if property_panel is not None:
        prop_x, prop_y, prop_x_label, prop_y_label, prop_log_x = property_panel
        ax = axes[-1]
        x = prop_x.reindex(perf_df.index).to_numpy(dtype=float)
        y = prop_y.reindex(perf_df.index).to_numpy(dtype=float)
        ax.scatter(x, y, s=26, color=UNCHANGED, alpha=0.75, edgecolors=SURFACE,
                   linewidths=0.8, zorder=3)
        # fitted and correlated over the analysed cohorts, the same set the performance
        # panels fit, so all three panels describe one population
        fit_x = np.log10(x) if prop_log_x else x
        ok = np.isfinite(fit_x) & np.isfinite(y)
        if ok.sum() >= 3:
            slope, intercept = np.polyfit(fit_x[ok], y[ok], 1)
            grid = np.linspace(fit_x[ok].min(), fit_x[ok].max(), 100)
            ax.plot(10 ** grid if prop_log_x else grid, slope * grid + intercept,
                    color=UNCHANGED, linewidth=1.4, alpha=0.9, zorder=2)
            # the correlation is the whole point of the panel, so state it rather than
            # leaving it to be eyeballed off the slope
            r = np.corrcoef(fit_x[ok], y[ok])[0, 1]
            ax.annotate(f"r = {r:.2f}  (n = {int(ok.sum())})", (0.99, 0.98),
                        xycoords="axes fraction", ha="right", va="top",
                        fontsize=8, color=INK_SECONDARY)
        if prop_log_x:
            ax.set_xscale("log")
        ax.set_xlabel(prop_x_label, color=INK_SECONDARY, fontsize=9)
        ax.set_ylabel(prop_y_label, color=INK_SECONDARY, fontsize=9)
        ax.grid(axis="both", color=GRIDLINE, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
        for spine in ax.spines.values():
            spine.set_color(AXIS)
            spine.set_linewidth(0.8)

    handles = [plt.Line2D([], [], marker="o", linestyle="-", markersize=7, linewidth=1.4,
                          color=REPRESENTATION_COLOUR[key], markeredgecolor=SURFACE, label=label)
               for key, label in series]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=9, labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, -0.06))
    _place_header(fig, title, subtitle)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved performance scatter to {path}.")


# Most cells sit near zero while a few labs move most of the way across the ranking, and on
# a plain linear ramp that tail washes the rest of the grid out to white. The ramp is
# stretched instead: shifts up to an emphasis point get EMPHASIS_FRACTION of each half of
# the colour range, and the tail beyond it shares what is left.
#
# The emphasis point is taken from the data, as the EMPHASIS_QUANTILE-th percentile of the
# absolute shifts in that figure, rather than being a fixed number of rank places. A fixed
# one cannot work: it has to stay proportionate both to how many features are ranked (which
# the blacklist changes) and to the figure's own spread, and the two rank-difference plots
# do not share a spread -- continuous-to-discretised has a median shift of about 5 places
# against a maximum near 50, continuous-to-binarised a median near 35 against a maximum
# past 120. Taking the percentile gives each figure a ramp in which EMPHASIS_QUANTILE% of
# its cells occupy EMPHASIS_FRACTION of the colour range, whatever the scale.
EMPHASIS_QUANTILE = 90
EMPHASIS_FRACTION = 0.85


def _emphasised_diverging_norm(limit: float, emphasis: float,
                               fraction: float = EMPHASIS_FRACTION) -> FuncNorm:
    """
    Symmetric norm over [-limit, limit], white at 0, that spends `fraction` of each half of
    the colour ramp on the first `emphasis` units of shift and the rest on everything above.
    Piecewise linear, so the colourbar stays readable and reversible.
    """
    knee = min(emphasis, limit)     # where the ramp changes slope, in data units
    tail = limit - knee             # data range left above the knee
    head = fraction if tail > 0 else 1.0  # colour range given to the part below the knee

    # only ufuncs, so masked cells stay masked on the way through
    def forward(x):
        magnitude = np.abs(x)
        scaled = head * np.minimum(magnitude, knee) / knee
        if tail > 0:
            scaled = scaled + (1 - head) * np.minimum(np.maximum(magnitude - knee, 0), tail) / tail
        return np.sign(x) * scaled

    def inverse(y):
        magnitude = np.abs(y)
        scaled = knee * np.minimum(magnitude, head) / head
        if tail > 0:
            scaled = scaled + tail * np.minimum(np.maximum(magnitude - head, 0), 1 - head) / (1 - head)
        return np.sign(y) * scaled

    return FuncNorm((forward, inverse), vmin=-limit, vmax=limit)


def plot_rank_difference_heatmap(diff_df: pd.DataFrame, title: str, subtitle: str,
                                 colourbar_label: str, path, cap: int | None = None,
                                 separate_last_row: bool = False) -> None:
    """
    One grid of signed rank shifts: rows are cohorts, columns are labs, cell colour is
    how far a lab moves in the importance ranking between two representations
    (rank_left - rank_right).
    """
    values = diff_df.to_numpy(dtype=float)
    limit = cap if cap is not None else np.nanmax(np.abs(values))
    if not np.isfinite(limit) or limit == 0:
        limit = 1.0  # degenerate all-equal/empty case: keep a valid symmetric range
    # the emphasis point comes from this figure's own spread, so the two rank-difference
    # plots each get a ramp matched to their very different distributions
    magnitudes = np.abs(values[np.isfinite(values)])
    emphasis = float(np.percentile(magnitudes, EMPHASIS_QUANTILE)) if magnitudes.size else 0.0
    if not np.isfinite(emphasis) or emphasis <= 0:
        emphasis = limit  # every cell identical: a plain linear ramp is all that is left
    norm = _emphasised_diverging_norm(float(limit), emphasis)

    # a blank spacer row before the last row
    row_labels = [str(i) for i in diff_df.index]
    if separate_last_row and len(values) > 1:
        values = np.insert(values, len(values) - 1, np.full(values.shape[1], np.nan), axis=0)
        row_labels.insert(len(row_labels) - 1, "")

    #  ramp  of red and blue with a white midpoint pinned at 0
    cmap = LinearSegmentedColormap.from_list("thresh_diverging", [LOST, "#ffffff", GAINED])
    cmap.set_bad(SURFACE)  # labs absent from a cohort disappear into the background

    # draw the mesh
    n_rows, n_cols = values.shape
    fig, ax = plt.subplots(figsize=(0.26 * n_cols + 4.0, 0.32 * n_rows + 3.0),
                           facecolor=SURFACE)
    ax.pcolormesh(np.ma.masked_invalid(values), cmap=cmap, norm=norm,
                  edgecolors=SURFACE, linewidth=1.0)  # surface gap between cells

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_xticklabels([_shorten(str(c)) for c in diff_df.columns], fontsize=6,
                       color=INK_SECONDARY, rotation=90)
    ax.set_yticklabels(row_labels, fontsize=9, color=INK_SECONDARY)
    ax.invert_yaxis()  # first cohort at the top
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colourbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                             ax=ax, fraction=0.02, pad=0.02)
    # ticks at the stretch points, so the uneven spacing of the ramp is stated outright
    knee = round(emphasis)
    ticks = sorted({t for t in (-limit, -knee, 0, knee, limit) if abs(t) <= limit})
    colourbar.set_ticks(ticks)
    colourbar.set_ticklabels([f"{t:.0f}" for t in ticks])
    colourbar.set_label(colourbar_label, color=INK_SECONDARY, fontsize=9)
    colourbar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    colourbar.outline.set_visible(False)

    _place_header(fig, title, subtitle)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved rank difference heatmap to {path}.")

# ---------- BLACKLIST COMPOSITION -----------------------------------------------------------------

# a lab item is one thing and a measurement is another, and the whole point of the figure
# is that the two do not rank the same way, so they get one panel each rather than one
# axis each on a shared plot
BLACKLIST_ITEM_FILL = UNCHANGED  # single series, no identity to carry -> neutral
BAR_HEIGHT = 0.34  # thin marks: a saturated fill this wide would read as a block


def _compact_count(value: float, _pos=None) -> str:
    """Axis tick formatter: 2500000 -> '2.5M'. Keeps the axis off scientific notation,
    whose shared '1e6' exponent sits away from the ticks and is easy to misread."""
    for cutoff, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= cutoff:
            scaled = value / cutoff
            return f"{scaled:.0f}{suffix}" if scaled == int(scaled) else f"{scaled:.1f}{suffix}"
    return f"{value:.0f}"


def plot_blacklist_composition(rows: list[tuple[str, int, int, int]], title: str,
                               subtitle: str, path, total_measurements: int,
                               note: str | None = None) -> None:
    """
    Two panels over the same rows (one row per exclusion reason): how many lab items each
    reason removes, and how many recorded measurements go with them, split into values that
    parse as numbers and values that do not.
    :param rows: list of (reason label, n_itemids, n_numeric_values, n_string_values)
    :param total_measurements: all non-NaN values in labevents, for the share labels
    """
    labels = [r[0] for r in rows]
    items = np.array([r[1] for r in rows], dtype=float)
    numeric = np.array([r[2] for r in rows], dtype=float)
    string = np.array([r[3] for r in rows], dtype=float)
    y = np.arange(len(rows))

    fig, (ax_items, ax_values) = plt.subplots(
        1, 2, figsize=(12.4, 0.72 * len(rows) + 3.2), facecolor=SURFACE, sharey=True,
        gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.08})

    # ----- left: lab items removed --------------------------------------------------
    ax_items.barh(y, items, height=BAR_HEIGHT, color=BLACKLIST_ITEM_FILL, zorder=2)
    for row, value in zip(y, items):
        ax_items.annotate(f"{value:,.0f}", (value, row), xytext=(6, 0),
                          textcoords="offset points", va="center", fontsize=9,
                          color=INK_PRIMARY)
    ax_items.set_xlabel("lab items", color=INK_SECONDARY, fontsize=9)

    # ----- right: measurements removed, split numeric / non-numeric -----------------
    # the surface gap between the two segments, so the boundary is visible without a rule
    ax_values.barh(y, numeric, height=BAR_HEIGHT, color=SERIES[0], zorder=2,
                   edgecolor=SURFACE, linewidth=1.5)
    ax_values.barh(y, string, left=numeric, height=BAR_HEIGHT, color=SERIES[1], zorder=2,
                   edgecolor=SURFACE, linewidth=1.5)
    for row, (num, txt) in enumerate(zip(numeric, string)):
        total = num + txt
        share = total / total_measurements * 100 if total_measurements else 0.0
        ax_values.annotate(f"{total:,.0f}  ({share:.1f}%)", (total, row), xytext=(6, 0),
                           textcoords="offset points", va="center", fontsize=9,
                           color=INK_PRIMARY)
    ax_values.set_xlabel("recorded measurements", color=INK_SECONDARY, fontsize=9)
    ax_values.xaxis.set_major_formatter(FuncFormatter(_compact_count))

    for ax in (ax_items, ax_values):
        ax.set_yticks(y)
        ax.set_ylim(len(rows) - 0.5, -0.5)  # first reason at the top
        ax.grid(axis="x", color=GRIDLINE, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
        # headroom for the direct labels, which sit outside the bar
        ax.set_xlim(0, max(ax.get_xlim()[1] * 1.18, 1))
        for side, spine in ax.spines.items():
            spine.set_visible(side == "left")
            spine.set_color(AXIS)
            spine.set_linewidth(0.8)
    ax_items.set_yticklabels(labels, fontsize=9, color=INK_SECONDARY)

    # two segments, so a legend is required; the segment colours are the categorical
    # slots in their fixed order
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                          markerfacecolor=colour, markeredgecolor=SURFACE, label=label)
               for colour, label in ((SERIES[0], "values that parse as numbers"),
                                     (SERIES[1], "values that do not"))]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9,
               labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, -0.02))
    if note is not None:
        fig.text(0.01, -0.07, _wrap_to_width(note, fig.get_figwidth()),
                 color=INK_MUTED, fontsize=8, ha="left", va="top")
    _place_header(fig, title, subtitle)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved blacklist composition to {path}.")


# ---------- COUNT PANELS ---------------------------------------------------------------------------

def plot_count_panels(panels: list[tuple[str, str, list[tuple[str, int]]]], title: str,
                      subtitle: str, path, note: str | None = None,
                      emphasis_label: str | None = None,
                      quiet_label: str | None = None) -> None:
    """
    Side by side panels of labelled counts, drawn as horizontal bars with the value direct
    labelled. Rows carry an emphasis flag rather than a hue each: the rows that answer the
    question take the accent, the rest recede to neutral, so the figure highlights instead
    of colouring five categories that the row labels already name.
    :param panels: list of (panel heading, x axis label, rows), rows = (label, value, emphasised)
    :param emphasis_label: legend text for the accented rows; omit for no legend
    :param quiet_label: legend text for the neutral rows
    """
    heights = [len(rows) for _, _, rows in panels]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(6.6 * len(panels), 0.46 * max(heights) + 3.4),
        facecolor=SURFACE, gridspec_kw={"wspace": 0.32})
    axes = np.atleast_1d(axes)

    for ax, (heading, x_label, rows) in zip(axes, panels):
        y = np.arange(len(rows))
        values = np.array([v for _, v, _ in rows], dtype=float)
        colours = [ACCENT if emphasised else UNCHANGED for _, _, emphasised in rows]
        ax.barh(y, values, height=BAR_HEIGHT, color=colours, zorder=2)
        for row, value in zip(y, values):
            ax.annotate(f"{value:,.0f}", (value, row), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9,
                        color=INK_PRIMARY)
        ax.set_yticks(y)
        ax.set_yticklabels([label for label, _, _ in rows], fontsize=9, color=INK_SECONDARY)
        ax.set_ylim(len(rows) - 0.5, -0.5)  # first row at the top
        ax.set_xlabel(x_label, color=INK_SECONDARY, fontsize=9)
        ax.set_title(heading, color=INK_PRIMARY, fontsize=10, loc="left", pad=10)
        ax.grid(axis="x", color=GRIDLINE, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
        ax.set_xlim(0, max(ax.get_xlim()[1] * 1.18, 1))  # headroom for the direct labels
        for side, spine in ax.spines.items():
            spine.set_visible(side == "left")
            spine.set_color(AXIS)
            spine.set_linewidth(0.8)

    if emphasis_label is not None:
        handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                              markerfacecolor=colour, markeredgecolor=SURFACE, label=label)
                   for colour, label in ((ACCENT, emphasis_label), (UNCHANGED, quiet_label))]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9,
                   labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, -0.02))
    if note is not None:
        fig.text(0.01, -0.08, _wrap_to_width(note, fig.get_figwidth()),
                 color=INK_MUTED, fontsize=8, ha="left", va="top")
    _place_header(fig, title, subtitle)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved count panels to {path}.")


# ---------- FAIRNESS PLOT: MODEL PERFORMANCE PER DEMOGRAPHIC GROUP ---------------------------------

NEG_FILL = "#c3c2b7"  # colour for negative samples
POS_FILL = "#e34948"  # colour for positive samples


def plot_fairness(panels: list[tuple[str, pd.DataFrame]], separator_after: list[str],
                  title: str, subtitle: str, path, share_x: bool = False,
                  x_min: float | None = None, dashed_after: list[str] | None = None,
                  counts_df: pd.DataFrame | None = None, note: str | None = None,
                  colours: dict[str, str] | None = None) -> None:
    """
    One panel per measure: rows are demographic groups, dots are representations.
    :param colours: column label -> colour. Without it the columns take SERIES in order,
        which only stretches to four and ties a colour to a column's position rather than
        to what it is; pass a map to give every representation the same colour it has
        everywhere else, and to plot more than four of them.
    """
    dashed_after = dashed_after or []
    groups = list(panels[0][1].index) # y-axis rows
    representations = list(panels[0][1].columns) # dots
    colour = dict(colours) if colours else dict(zip(representations, SERIES))
    missing = [rep for rep in representations if rep not in colour]
    if missing:
        raise ValueError(f"no colour for {missing}; pass colours= for more than "
                         f"{len(SERIES)} representations")
    offsets = np.linspace(-0.26, 0.26, len(representations))

    # helper to create row separators solid or dashed
    def _row_separators(ax):
        for row, group in enumerate(groups):
            if group in separator_after:
                ax.axhline(row + 0.5, color=AXIS, linewidth=0.8, zorder=1)
            elif group in dashed_after:
                ax.axhline(row + 0.5, color=AXIS, linewidth=0.8, linestyle="--", zorder=1)
    # layout
    n_cols = len(panels) + (1 if counts_df is not None else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.4 * n_cols + 1.6,
                                                 0.42 * len(groups) + 2.6),
                             sharey=True, sharex=False, facecolor=SURFACE,
                             gridspec_kw={"wspace": 0.18})
    axes = np.atleast_1d(axes)
    measure_axes = axes[:len(panels)]
    # if possible make x-axis scale comparable
    if share_x:
        for ax in measure_axes[1:]:
            ax.sharex(measure_axes[0])

    # draw measure panels
    for ax, (measure_label, panel_df) in zip(measure_axes, panels):
        for row, group in enumerate(groups):
            for representation, offset in zip(representations, offsets):
                value = panel_df.loc[group, representation]
                if pd.isna(value):
                    continue
                ax.plot(value, row + offset, marker="o", markersize=7,
                        color=colour[representation], markeredgecolor=SURFACE,
                        markeredgewidth=2, zorder=3)
        _row_separators(ax)

        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        if x_min is not None:
            ax.set_xlim(left=x_min)
        ax.set_ylim(len(groups) - 0.5, -0.5)  # first group at the top
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups, fontsize=9, color=INK_SECONDARY)
        ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
        ax.set_title(measure_label, color=INK_PRIMARY, fontsize=11, pad=10)

    # draw counts panel
    if counts_df is not None:
        cax = axes[-1]
        counts = counts_df.reindex(groups)  # align to the row order, blanks -> NaN
        neg = counts["n_negative"].to_numpy(dtype=float)
        pos = counts["n_positive"].to_numpy(dtype=float)
        for row in range(len(groups)):
            if np.isnan(neg[row]):
                continue  # group without counts (e.g. the average row)
            total = neg[row] + pos[row]
            cax.barh(row, neg[row], height=0.62, color=NEG_FILL, zorder=3)
            cax.barh(row, pos[row], left=neg[row], height=0.62, color=POS_FILL,
                     zorder=3, edgecolor=SURFACE, linewidth=0.5)
            prevalence = pos[row] / total if total else 0.0
            cax.annotate(f"{int(total):,} ({int(pos[row]):,} pos, {prevalence:.1%})",
                         (total, row), xytext=(5, 0), textcoords="offset points",
                         ha="left", va="center", fontsize=8, color=INK_SECONDARY)
        _row_separators(cax)

        cax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            cax.spines[side].set_visible(False)
        cax.spines["bottom"].set_color(AXIS)
        cax.spines["bottom"].set_linewidth(0.8)
        cax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
        cax.set_axisbelow(True)
        cax.set_xlim(0, np.nanmax(neg + pos) * 1.3)  # headroom for the count labels
        cax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
        cax.set_title("Sample count", color=INK_PRIMARY, fontsize=11, pad=10)
        class_handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=9,
                                    color=NEG_FILL, label="negative"),
                         plt.Line2D([], [], marker="s", linestyle="none", markersize=9,
                                    color=POS_FILL, label="positive")]
        cax.legend(handles=class_handles, loc="lower right", frameon=False,
                   fontsize=8, labelcolor=INK_SECONDARY)

    # the group labels are short, so the default left margin leaves a wide empty gap
    # that pushes the panels to the right; tighten both margins to centre the block
    fig.subplots_adjust(left=0.045, right=0.995)

    handles = [plt.Line2D([], [], linestyle="none", marker="o", markersize=8,
                          color=colour[representation], label=representation)
               for representation in representations]
    fig.legend(handles=handles, loc="lower center", ncol=len(representations),
               frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
               bbox_to_anchor=(0.5, -0.02))
    _place_header(fig, title, subtitle)
    if note:
        # footnote under the legend; tight bbox grows to include it
        fig.text(0.01, -0.06, note, color=INK_MUTED, fontsize=8, ha="left")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved fairness plot to {path}.")
