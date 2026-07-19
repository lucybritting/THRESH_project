import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
GAINED = "#2a78d6"     # moved towards rank 1
LOST = "#e34948"       # moved away from rank 1
UNCHANGED = "#898781"  # neutral midpoint

# categorical slots for highlighting individual features, in fixed order. Never
# cycled: past four, the caller should highlight fewer rather than reuse a hue.
SERIES = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]
ACCENT = "#2a78d6"     # single accent: "in this panel's top group", not an identity

# ordinal ramp for ranks: one hue, evenly spaced steps, light end still visible
# against the surface (validated with validate_palette.py --ordinal).
ORDINAL_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]

# sequential ramp for magnitudes: one hue, light -> dark. Never a rainbow.
SEQUENTIAL_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
                   "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
                   "#0d366b"]

MAX_LABEL_CHARS = 20


def _shorten(name: str) -> str:
    return name if len(name) <= MAX_LABEL_CHARS else name[:MAX_LABEL_CHARS - 1] + "…"


def _slope_panel(ax, panel_df: pd.DataFrame, left_label: str, right_label: str,
                 ylim: tuple[float, float]) -> None:
    """
    Draw one slope chart: a line per row of panel_df, from its rank under the left
    representation to its rank under the right one.
    :param panel_df: columns label, rank_left, rank_right
    :param ylim: shared across panels, so equal slopes mean equal rank shifts
    """
    # recessive hairline guides: one horizontal rule per occupied rank, and a
    # vertical rule down each column of dots. Drawn first so the slopes sit on top.
    occupied_ranks = pd.unique(pd.concat([panel_df["rank_left"], panel_df["rank_right"]]))
    ax.hlines(occupied_ranks, xmin=0, xmax=1, color=GRIDLINE, linewidth=0.8, zorder=1)
    for x in (0, 1):
        ax.axvline(x, color=GRIDLINE, linewidth=0.8, zorder=1)

    for label, rank_left, rank_right in zip(panel_df["label"], panel_df["rank_left"],
                                            panel_df["rank_right"]):
        if rank_right == rank_left:
            colour = UNCHANGED
        else:
            colour = GAINED if rank_right < rank_left else LOST  # smaller rank = more important
        ax.plot([0, 1], [rank_left, rank_right], color=colour, linewidth=1.6,
                marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2,
                zorder=3)  # surface ring keeps overlapping markers separated
        # label both ends: the reader should never have to trace a line back to a legend
        ax.annotate(f"{label}  {rank_left:.0f}", (0, rank_left), xytext=(-10, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=8, color=INK_SECONDARY)
        ax.annotate(f"{rank_right:.0f}  {label}", (1, rank_right), xytext=(10, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=8, color=INK_SECONDARY)

    ax.set_ylim(*ylim)  # rank 1 at the top
    ax.set_xlim(-0.62, 1.62)

    # the two labelled columns are the axis: no box, no grid, no y ticks
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
    # a single rank scale across the panels, otherwise equal slopes would mean
    # different shifts in each panel and the two could not be compared
    all_ranks = pd.concat([panel_df[["rank_left", "rank_right"]] for _, _, panel_df in panels])
    lo, hi = all_ranks.min().min(), all_ranks.max().max()
    pad = 0.04 * (hi - lo)
    ylim = (hi + pad, lo - pad)

    # wspace leaves room for the labels hanging off both sides of each panel,
    # otherwise one panel's right-hand labels collide with the next one's left
    fig, axes = plt.subplots(1, len(panels), figsize=(7.3 * len(panels), 8.4),
                             facecolor=SURFACE, gridspec_kw={"wspace": 0.75})
    for ax, (left_label, right_label, panel_df) in zip(np.atleast_1d(axes), panels):
        _slope_panel(ax, panel_df, left_label, right_label, ylim)

    handles = [plt.Line2D([], [], color=GAINED, linewidth=1.6, label="gained importance"),
               plt.Line2D([], [], color=LOST, linewidth=1.6, label="lost importance"),
               plt.Line2D([], [], color=UNCHANGED, linewidth=1.6, label="unchanged")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9, labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, color=INK_PRIMARY, fontsize=13, x=0.01, y=1.01, ha="left")
    fig.text(0.01, 0.96, subtitle, color=INK_MUTED, fontsize=9, ha="left")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved rank shift slope chart to {path}.")


def _bump_panel(ax, panel_df: pd.DataFrame, panel_title: str, highlighted: list,
                ylim: tuple[float, float]) -> None:
    """
    Draw one bump chart: a line per row of panel_df, tracing that feature's rank
    across the folds.
    :param panel_df: index = feature label, columns = folds, values = rank (1 = most important)
    :param highlighted: labels to draw in the accent colour and label directly.
                        Each panel highlights its own features, so the accent means
                        "top of this representation" -- one meaning in every panel.
                        Identity is carried by the end label, never by the colour.
    """
    folds = panel_df.columns.to_numpy()
    for label, ranks in panel_df.iterrows():
        if label in highlighted:
            continue  # drawn afterwards, on top of the grey context
        ax.plot(folds, ranks.to_numpy(), color=INK_MUTED, linewidth=1.0,
                alpha=0.35, zorder=2)
    for label in highlighted:
        if label not in panel_df.index:
            continue
        ranks = panel_df.loc[label].to_numpy()
        ax.plot(folds, ranks, color=ACCENT, linewidth=2.0, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=2,
                zorder=3)  # surface ring keeps overlapping markers separated
        ax.annotate(f" {_shorten(str(label))}", (folds[-1], ranks[-1]),
                    xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=8, color=INK_SECONDARY, annotation_clip=False)

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
    how each feature's importance rank moves across the folds. Flat lines mean the
    folds agree on the ordering, crossings mean they do not.
    :param panels: (panel_title, panel_df, highlighted) triples. panel_df is indexed
                   by feature label, one column per fold, values are ranks.
                   highlighted are that panel's own labels to accent and name.
    :param title: figure heading
    :param subtitle: one line under the heading, e.g. what was selected
    :param path: pathlib.Path the png is written to
    """
    all_ranks = pd.concat([panel_df for _, panel_df, _ in panels])
    ylim = (all_ranks.to_numpy().max() + 0.5, 0.5)

    # wspace leaves room for the labels hanging off the right of each panel
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 7.6),
                             sharey=True, facecolor=SURFACE,
                             gridspec_kw={"wspace": 0.85})
    for ax, (panel_title, panel_df, highlighted) in zip(np.atleast_1d(axes), panels):
        _bump_panel(ax, panel_df, panel_title, highlighted, ylim)

    first = np.atleast_1d(axes)[0]
    first.set_ylabel("importance rank (1 = most important)", color=INK_SECONDARY, fontsize=9)

    fig.suptitle(title, color=INK_PRIMARY, fontsize=13, x=0.02, y=1.02, ha="left")
    fig.text(0.02, 0.965, subtitle, color=INK_MUTED, fontsize=9, ha="left")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved rank bump chart to {path}.")


def plot_importance_rank_heatmap(rank_df: pd.DataFrame, group_sizes: dict, title: str,
                                 subtitle: str, colourbar_label: str, path,
                                 rank_cap: int = 40) -> None:
    """
    One grid: rows are (cohort, representation), columns are features, cell colour
    is the feature's importance rank -- dark = most important.

    Ranks rather than raw importances, because importances are shares that depend
    on how many features a cohort has, and because half the raw values would sit in
    the bottom quarter of the ramp. Ranks also match the rest of the analysis.

    :param rank_df: rows indexed by representation label, one column per feature
    :param group_sizes: cohort -> number of consecutive rows it owns, in row order.
                        A blank row is drawn between groups so a cohort's
                        representations read as one block.
    :param rank_cap: ranks past this share the lightest step ("outside the top N here")
    """
    # interleave the cohort blocks with blank spacer rows: NaN renders as the surface
    matrix, row_labels, group_centres, row, start = [], [], {}, 0, 0
    for index, (group, size) in enumerate(group_sizes.items()):
        if index:
            matrix.append(np.full(rank_df.shape[1], np.nan))
            row_labels.append("")
            row += 1
        group_centres[group] = row + size / 2
        for offset in range(size):
            matrix.append(rank_df.to_numpy()[start + offset])
            row_labels.append(rank_df.index[start + offset])
        start += size
        row += size
    matrix = np.array(matrix)

    cmap = LinearSegmentedColormap.from_list("thresh_rank", ORDINAL_BLUE[::-1])
    cmap.set_bad(SURFACE)  # spacer rows disappear into the background

    n_rows, n_features = matrix.shape
    fig, ax = plt.subplots(figsize=(0.42 * n_features + 6.0, 0.36 * n_rows + 2.4),
                           facecolor=SURFACE)
    ax.pcolormesh(np.ma.masked_invalid(matrix), cmap=cmap, vmin=1, vmax=rank_cap,
                  edgecolors=SURFACE, linewidth=1.5)  # surface gap between cells

    ax.set_xticks(np.arange(n_features) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_xticklabels([_shorten(str(c)) for c in rank_df.columns], fontsize=7,
                       color=INK_SECONDARY, rotation=45, ha="right")
    ax.set_yticklabels(row_labels, fontsize=8, color=INK_SECONDARY)
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # cohort names sit outside the representation labels, naming each block once
    for group, centre in group_centres.items():
        # offset clears the widest representation label, so the two never collide
        ax.annotate(group, xy=(0, centre), xycoords=("axes fraction", "data"),
                    xytext=(-205, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=9.5, color=INK_PRIMARY, annotation_clip=False)

    colourbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap,
                                                   norm=plt.Normalize(1, rank_cap)),
                             ax=ax, fraction=0.02, pad=0.02)
    colourbar.set_label(colourbar_label, color=INK_SECONDARY, fontsize=9)
    colourbar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    colourbar.ax.invert_yaxis()  # rank 1 at the top, matching "most important first"
    colourbar.set_ticks([1, 10, 20, 30, rank_cap])
    colourbar.set_ticklabels(["1", "10", "20", "30", f"{rank_cap}+"])
    colourbar.outline.set_visible(False)

    fig.suptitle(title, color=INK_PRIMARY, fontsize=13, x=0.01, y=1.04, ha="left")
    fig.text(0.01, 1.0, subtitle, color=INK_MUTED, fontsize=9, ha="left")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved feature importance rank heatmap to {path}.")


def plot_kendalls_w_heatmap(w_df: pd.DataFrame, title: str, subtitle: str,
                            colourbar_label: str, path) -> None:
    """
    Heatmap of Kendall's W: rows are cohorts, columns are representations.
    W is a single number per (cohort, representation) -- it scores the agreement
    over the whole feature ranking at once -- so there is no per-feature column.
    :param w_df: indexed by cohort, one column per representation, values are W
    :param title: figure heading
    :param subtitle: one line under the heading
    :param colourbar_label: what the colour encodes
    :param path: pathlib.Path the png is written to
    """
    values = w_df.to_numpy()
    # W sits in a narrow band near 1, so the ramp spans the observed range rather
    # than 0-1: on a full scale every cell would read as the same blue. Each cell
    # carries its value, so the exact numbers never depend on reading the colour.
    vmin, vmax = values.min(), values.max()
    cmap = LinearSegmentedColormap.from_list("thresh_blue", SEQUENTIAL_BLUE)

    n_rows, n_cols = w_df.shape
    fig, ax = plt.subplots(figsize=(1.9 * n_cols + 3.4, 0.72 * n_rows + 2.2),
                           facecolor=SURFACE)
    ax.pcolormesh(values, cmap=cmap, vmin=vmin, vmax=vmax,
                  edgecolors=SURFACE, linewidth=2)  # surface gap between cells

    # a value in every cell: 12 cells is a table, and the colour only ranks them
    midpoint = (vmin + vmax) / 2
    for row in range(n_rows):
        for col in range(n_cols):
            value = values[row, col]
            ax.annotate(f"{value:.3f}", (col + 0.5, row + 0.5), ha="center", va="center",
                        fontsize=9, color=SURFACE if value > midpoint else INK_PRIMARY)

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_xticklabels(w_df.columns, fontsize=9, color=INK_SECONDARY)
    ax.set_yticklabels(w_df.index, fontsize=9, color=INK_SECONDARY)
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

    fig.suptitle(title, color=INK_PRIMARY, fontsize=13, x=0.01, y=1.06, ha="left")
    fig.text(0.01, 1.0, subtitle, color=INK_MUTED, fontsize=9, ha="left")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved Kendall's W heatmap to {path}.")