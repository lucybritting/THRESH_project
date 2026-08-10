import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import pandas as pd

# colours: the diverging pair blue <-> red with a neutral grey midpoint, validated
# for colour blindness on the light surface. The polarity being encoded is the
# direction of the rank shift; every line is also direct-labelled, so the colour
# only reinforces what the labels and the slope already show.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"   # hairline guides, one shade off the surface
AXIS = "#c3c2b7"       # baseline / separator rules, a step darker than the grid
GAINED = "#2a78d6"     # moved towards rank 1
LOST = "#e34948"       # moved away from rank 1
UNCHANGED = "#898781"  # neutral midpoint

# categorical slots for highlighting individual features, in fixed order. Never
# cycled: past four, the caller should highlight fewer rather than reuse a hue.
SERIES = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]
ACCENT = "#2a78d6"     # single accent: "in this panel's top group", not an identity
DUMBBELL_RIGHT = "#eda100"  # second endpoint of a dumbbell, paired with ACCENT

# two flat class colours for the classification heatmap: negatives recede, positives
# carry the accent, so a row of predictions reads as "where did it say positive"
CLASS_NEG = "#dbe4ec"  # label / prediction 0
CLASS_POS = "#2a78d6"  # label / prediction 1

# a representation always gets the same colour across the fairness figures, matching
# the SERIES order used by plot_fairness (cont, disc, disc_imp, bin)
REPRESENTATION_COLOUR = {"cont": SERIES[0], "disc": SERIES[1],
                         "disc_imp": SERIES[2], "bin": SERIES[3]}

# sequential ramp for magnitudes: one hue, light -> dark. Never a rainbow.
SEQUENTIAL_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
                   "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
                   "#0d366b"]

MAX_LABEL_CHARS = 20


def _shorten(name: str) -> str:
    return name if len(name) <= MAX_LABEL_CHARS else name[:MAX_LABEL_CHARS - 1] + "…"


# the title block is placed at fixed inch offsets from the top of the figure, and a
# fixed strip is reserved above the axes, so both the header-subheader gap AND the
# distance from the block to the content stay the same on every plot, whatever its size
HEADER_TOP_IN = 0.30            # header baseline below the figure's top edge
HEADER_SUBHEADER_GAP_IN = 0.30  # gap between header and subheader
HEADER_STRIP_IN = 1.05          # top strip reserved for the block; content starts below it


def _place_header(fig, title: str, subtitle: str) -> None:
    """
    Top-left header and subheader placed at fixed inch offsets from the figure top,
    with a fixed strip reserved above the axes. All offsets are converted to figure
    fractions via the figure height, so the spacing does not scale with the figure.
    """
    h = fig.get_figheight()
    fig.subplots_adjust(top=1 - HEADER_STRIP_IN / h)  # reserve the title strip
    fig.suptitle(title, color=INK_PRIMARY, fontsize=13, x=0.01,
                 y=1 - HEADER_TOP_IN / h, ha="left")
    fig.text(0.01, 1 - (HEADER_TOP_IN + HEADER_SUBHEADER_GAP_IN) / h, subtitle,
             color=INK_MUTED, fontsize=9, ha="left")

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
                       separate_last_row: bool = False) -> None:
    """
    Heatmap of a value (Kendalls W or performance measures)
    """
    values = df.to_numpy(dtype=float)
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
            ax.annotate(f"{value:.3f}", (col + 0.5, row + 0.5), ha="center", va="center",
                        fontsize=9, color=SURFACE if value > midpoint else INK_PRIMARY) # text colour flips depending on cell colour
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
    fig, ax = plt.subplots(figsize=(0.26 * n_cols + 4.0, 0.5 * n_rows + 3.0),
                           facecolor=SURFACE)
    ax.pcolormesh(np.ma.masked_invalid(values), cmap=cmap, vmin=-limit, vmax=limit,
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

    colourbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap,
                                                   norm=plt.Normalize(-limit, limit)),
                             ax=ax, fraction=0.02, pad=0.02)
    colourbar.set_label(colourbar_label, color=INK_SECONDARY, fontsize=9)
    colourbar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    colourbar.outline.set_visible(False)

    _place_header(fig, title, subtitle)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved rank difference heatmap to {path}.")

# ---------- FAIRNESS PLOT: MODEL PERFORMANCE PER DEMOGRAPHIC GROUP ---------------------------------

NEG_FILL = "#c3c2b7"  # colour for negative samples
POS_FILL = "#e34948"  # colour for positive samples


def plot_fairness(panels: list[tuple[str, pd.DataFrame]], separator_after: list[str],
                  title: str, subtitle: str, path, share_x: bool = False,
                  x_min: float | None = None, dashed_after: list[str] | None = None,
                  counts_df: pd.DataFrame | None = None, note: str | None = None) -> None:
    """
    One panel per measure: rows are demographic groups, dots are representations.


    """
    dashed_after = dashed_after or []
    groups = list(panels[0][1].index) # y-axis rows
    representations = list(panels[0][1].columns) # dots
    colour = dict(zip(representations, SERIES))  # fixed order, never cycled
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