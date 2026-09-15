"""Fixed TOIAM penalty and manuscript geometry (dimensions supplied in TeX pt)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DELTA = 0.05
TEXTWIDTH_PT = 370.38374
TEXTHEIGHT_PT = 598.0
PAPERWIDTH_PT = 614.295
PAPERHEIGHT_PT = 794.96999
TEX_POINTS_PER_INCH = 72.27  # PDF/PostScript points instead use 72 per inch.
FIGURE_WIDTH_IN = TEXTWIDTH_PT / TEX_POINTS_PER_INCH
MAX_FIGURE_HEIGHT_IN = TEXTHEIGHT_PT / TEX_POINTS_PER_INCH
PAPER_SIZE_IN = (PAPERWIDTH_PT / TEX_POINTS_PER_INCH,
                 PAPERHEIGHT_PT / TEX_POINTS_PER_INCH)


def save_manuscript_pdf(fig, path):
    """Export an uncropped, text-width figure for inclusion in the manuscript."""
    width, height = fig.get_size_inches()
    if not np.isclose(width, FIGURE_WIDTH_IN, rtol=0, atol=1e-9):
        raise ValueError("Figure width must match the manuscript text width.")
    if not 0 < height <= MAX_FIGURE_HEIGHT_IN + 1e-9:
        raise ValueError("Figure height must fit within the manuscript text height.")
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        raise ValueError("Manuscript figures must use a .pdf filename.")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reset the global savefig bbox too: bbox_inches=None otherwise inherits it.
    with plt.rc_context({"savefig.bbox": None, "pdf.fonttype": 42}):
        fig.savefig(path, format="pdf", dpi=300, bbox_inches=None,
                    metadata={"Subject": "TOIAM experiment; delta=0.05"})
    return path


def interval_pages(total, selected, per_figure):
    """Group zero-based interval indices in time order, without omitting any."""
    if isinstance(per_figure, (bool, np.bool_)) or not isinstance(
        per_figure, (int, np.integer)
    ) or per_figure < 1:
        raise ValueError("OVERLAY_INTERVALS_PER_FIGURE must be a positive integer.")
    indices = list(range(total)) if selected is None else list(selected)
    if not indices or any(
        isinstance(k, (bool, np.bool_))
        or not isinstance(k, (int, np.integer)) or not 0 <= k < total
        for k in indices
    ):
        raise ValueError("OVERLAY_INTERVALS must contain valid zero-based indices.")
    if len(set(indices)) != len(indices):
        raise ValueError("OVERLAY_INTERVALS must not contain duplicates.")
    indices = sorted(indices)
    return [indices[first:first + per_figure]
            for first in range(0, len(indices), per_figure)]
