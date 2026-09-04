"""Colours, label shortening and the header block: the look every chart shares."""

import textwrap


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

# A representation always gets the same colour wherever representations are drawn as
# series. All seven feature sets are plotted, so this follows the categorical theme's own
# fixed hue order rather than the four SERIES slots -- seven is past what SERIES can carry.
# The order is validated on adjacent pairs, which is the criterion that applies when the
# series sit side by side as they do in the box plots. It does NOT pass on all pairs: no
# seven-colour subset of the palette does, so a plot that overlays all seven in one space
# (rather than ordering them) cannot rely on hue alone to separate them.
CONCATENATED = "#4a3aa7"
REPRESENTATION_COLOUR = {"cont": "#2a78d6",          # blue
                         "disc": "#eb6834",          # orange
                         "disc_imp": "#1baf7a",      # aqua
                         "bin": "#eda100",           # yellow
                         "cont_bin": "#e87ba4",      # magenta
                         "disc_bin": "#008300",      # green
                         "cont_disc_bin": CONCATENATED}  # violet

# sequential ramp for magnitudes: one hue, light -> dark. Never a rainbow.
SEQUENTIAL_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
                   "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
                   "#0d366b"]

# room enough for a lab name plus the "(fluid)" or "(category)" qualifier that
# load_labitem_labels() adds to labels shared by several itemids
MAX_LABEL_CHARS = 28


def _shorten(name: str) -> str:
    """
    Truncate a lab name for an axis. A trailing "(qualifier)" is the only thing telling two
    identically named labs apart, so it is kept whole and the name in front of it gives way
    instead -- truncating "Eosinophils (Pleural)" to "Eosinophils (Pleu…" would lose exactly
    the part that matters. If that leaves too little of the name to read, the qualifier goes
    after all and the label is cut plainly.
    """
    if len(name) <= MAX_LABEL_CHARS:
        return name
    head, separator, qualifier = name.rpartition(" (")
    if separator and qualifier.endswith(")"):
        room = MAX_LABEL_CHARS - len(separator) - len(qualifier)
        if room >= 6:
            return head[:room - 1] + "…" + separator + qualifier
    return name[:MAX_LABEL_CHARS - 1] + "…"


# the title block is placed at fixed inch offsets from the top of the figure, and a
# fixed strip is reserved above the axes, so both the header-subheader gap AND the
# distance from the block to the content stay the same on every plot, whatever its size
HEADER_TOP_IN = 0.30            # header baseline below the figure's top edge
HEADER_SUBHEADER_GAP_IN = 0.30  # gap between header and subheader
HEADER_STRIP_IN = 1.05          # top strip reserved for the block; content starts below it


SUBHEADER_FONTSIZE = 9
SUBHEADER_LINE_IN = SUBHEADER_FONTSIZE * 1.25 / 72  # line height, inches


def _wrap_to_width(text: str, fig_width_in: float, fontsize: float = 8) -> str:
    """
    Hard-wrap text to the figure width. Without this a long string is laid out as one
    unbroken line and `bbox_inches="tight"` grows the saved figure to fit it, which silently
    stretches the whole plot -- so the widest sentence, not the data, would set the figure
    size. A sans glyph averages a bit over half its point size in width.
    """
    chars_per_line = fig_width_in * 72 / (fontsize * 0.55)
    return textwrap.fill(text, width=max(40, int(chars_per_line)))


def _place_header(fig, title: str, subtitle: str) -> None:
    """
    Top-left header and subheader placed at fixed inch offsets from the figure top,
    with a strip reserved above the axes. All offsets are converted to figure fractions via
    the figure height, so the spacing does not scale with the figure.

    The subheader is wrapped to the figure width and the reserved strip grows by a line for
    each extra line of it. Otherwise a long subheader is one unbroken line and the tight
    bounding box stretches the figure to fit the sentence, which makes the prose rather than
    the data decide how wide the plot is.
    """
    h = fig.get_figheight()
    wrapped = _wrap_to_width(subtitle, fig.get_figwidth(), SUBHEADER_FONTSIZE)
    extra_lines = wrapped.count("\n")
    fig.subplots_adjust(top=1 - (HEADER_STRIP_IN + extra_lines * SUBHEADER_LINE_IN) / h)
    fig.suptitle(title, color=INK_PRIMARY, fontsize=13, x=0.01,
                 y=1 - HEADER_TOP_IN / h, ha="left")
    # anchored by its top edge, so extra lines grow downward into the strip reserved above
    fig.text(0.01, 1 - (HEADER_TOP_IN + HEADER_SUBHEADER_GAP_IN) / h, wrapped,
             color=INK_MUTED, fontsize=SUBHEADER_FONTSIZE, ha="left", va="top")
