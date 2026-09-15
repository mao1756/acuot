"""Combined source/momentum PDFs; run with %run -i after the TOIAM solves.

Also executed by the notebook. Reads method_data at the fixed delta=0.05;
does not run the optimizer. Plot options can be set in the calling kernel.
"""

from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, SymLogNorm
from matplotlib.ticker import FuncFormatter
from toiam_figures import (
    DELTA, FIGURE_WIDTH_IN, MAX_FIGURE_HEIGHT_IN,
    interval_pages, save_manuscript_pdf,
)

if globals().get("RESULTS_DELTA") != DELTA or not globals().get("method_data"):
    raise RuntimeError("Run the fixed-delta TOIAM WFR and ACUOT cells first.")

RAW_IMAGE_DIR = globals().get("RAW_IMAGE_DIR", env_path("TOIAM_RAW_DIR") or (TOIAM_ROOT / SEQUENCE))
OVERLAY_INTERVALS = globals().get("OVERLAY_INTERVALS", None)
OVERLAY_INTERVALS_PER_FIGURE = globals().get("OVERLAY_INTERVALS_PER_FIGURE", 2)
OVERLAY_PDF_DIR = globals().get("OVERLAY_PDF_DIR", REPO_ROOT / "paper" / "figures")
SOURCE_COLOR_LIMIT = globals().get("SOURCE_COLOR_LIMIT", None)
SOURCE_LINEAR_FRACTION = globals().get("SOURCE_LINEAR_FRACTION", 0.001)
SOURCE_MAX_ALPHA = globals().get("SOURCE_MAX_ALPHA", 0.65)
NEW_CELL_COLOR = "#00ffff"
QUIVER_STRIDE = globals().get("QUIVER_STRIDE", 2)
QUIVER_SCALE = globals().get("QUIVER_SCALE", None)
QUIVER_LOG_LENGTHS = globals().get("QUIVER_LOG_LENGTHS", True)

if isinstance(QUIVER_STRIDE, (bool, np.bool_)) or not isinstance(
    QUIVER_STRIDE, (int, np.integer)
) or QUIVER_STRIDE < 1:
    raise ValueError("QUIVER_STRIDE must be a positive integer.")
if not np.isfinite(SOURCE_MAX_ALPHA) or not 0 <= SOURCE_MAX_ALPHA <= 1:
    raise ValueError("SOURCE_MAX_ALPHA must lie in [0, 1].")
pages = interval_pages(T_STEPS, OVERLAY_INTERVALS, OVERLAY_INTERVALS_PER_FIGURE)

raw_image_files = index_tiffs(RAW_IMAGE_DIR)
missing_raw = sorted(set(map(int, node_frames)) - set(raw_image_files))
missing_tracking = sorted(set(map(int, node_frames)) - set(tracking_files))
if missing_raw:
    raise FileNotFoundError(
        f"Missing original microscope frames {missing_raw} in {RAW_IMAGE_DIR}. "
        "Set RAW_IMAGE_DIR or TOIAM_RAW_DIR to the original sequence TIFF directory."
    )
if missing_tracking:
    raise FileNotFoundError(f"Missing tracking masks for frames {missing_tracking}.")


def read_overlay_tiff(path):
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"Expected a single-page microscope image or mask: {path}")
        array = np.array(image)
    if array.ndim != 2 or array.shape != image_shape:
        raise ValueError(f"Expected image shape {image_shape}, got {array.shape}: {path}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Non-finite image values: {path}")
    return array


# Compare persistent tracking IDs, not frame-local segmentation IDs or pixels:
# cell motion alone must not be mistaken for cell birth.
microscope_frames = []
new_cell_masks = []
new_track_counts = []
previous_ids = None
for frame in node_frames:
    frame = int(frame)
    microscope_frames.append(read_overlay_tiff(raw_image_files[frame]))
    labels = read_overlay_tiff(tracking_files[frame])
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError(f"Expected nonnegative integer tracking labels at frame {frame}.")
    current_ids = np.unique(labels[labels > 0])
    expected_ids = {
        label for label, (begin, end, _) in tracks.items() if begin <= frame <= end
    }
    if set(current_ids.tolist()) != expected_ids:
        raise ValueError(f"Tracking mask/table mismatch at selected frame {frame}.")
    new_ids = (
        np.array([], dtype=current_ids.dtype)
        if previous_ids is None
        else np.setdiff1d(current_ids, previous_ids, assume_unique=True)
    )
    new_cell_masks.append(np.isin(labels, new_ids))
    new_track_counts.append(len(new_ids))
    previous_ids = current_ids

# Keep image contrast, spatial registration, and the signed source scale fixed
# across every frame and method. Images use the same top-left origin as
# the (y, x) solver grid; their common extent is in normalized image coordinates.
image_extent = (0, LX, LY, 0)
contrast_samples = np.concatenate([image.ravel()[::16] for image in microscope_frames])
image_vmin, image_vmax = np.percentile(contrast_samples, [1, 99])
if image_vmax <= image_vmin:
    image_vmin = float(min(image.min() for image in microscope_frames))
    image_vmax = float(max(image.max() for image in microscope_frames))
    if image_vmax <= image_vmin:
        image_vmax = image_vmin + 1.0

source_fields = {}
source_abs_max = 0.0
if not method_data:
    raise ValueError(f"No retained methods for delta={DELTA:g}.")
for name, (solution, _, _, _, _) in method_data.items():
    source = np.asarray(solution.V.Z)
    if source.shape != (T_STEPS, NY, NX) or not np.all(np.isfinite(source)):
        raise ValueError(f"Invalid centered source field for {name}, delta={DELTA:g}.")
    source_fields[name] = source
    source_abs_max = max(source_abs_max, float(np.max(np.abs(source))))

source_limit = (
    (source_abs_max if source_abs_max > 0 else 1.0)
    if SOURCE_COLOR_LIMIT is None
    else float(SOURCE_COLOR_LIMIT)
)
if not np.isfinite(source_limit) or source_limit <= 0:
    raise ValueError("SOURCE_COLOR_LIMIT must be a finite positive number or None.")
if not 0 < SOURCE_LINEAR_FRACTION <= 1:
    raise ValueError("SOURCE_LINEAR_FRACTION must lie in (0, 1].")
source_norm = SymLogNorm(
    linthresh=max(SOURCE_LINEAR_FRACTION * source_limit, np.finfo(float).tiny),
    vmin=-source_limit, vmax=source_limit, base=10,
)
source_cmap = plt.get_cmap("RdBu_r")
new_cell_cmap = ListedColormap([NEW_CELL_COLOR])
source_colorbar_extend = "both" if source_abs_max > source_limit else "neither"


def draw_microscope_overlay(ax, node_index, *, source=None, fill_new=False):
    ax.imshow(
        microscope_frames[node_index], cmap="gray", vmin=image_vmin, vmax=image_vmax,
        origin="upper", extent=image_extent, interpolation="nearest",
    )
    if source is not None:
        # Zero source is transparent; red is creation and blue is destruction.
        strength = np.clip(np.abs(2 * source_norm(source) - 1), 0, 1)
        alpha = SOURCE_MAX_ALPHA * np.sqrt(strength)
        ax.imshow(
            source, cmap=source_cmap, norm=source_norm, alpha=alpha,
            origin="upper", extent=image_extent, interpolation="nearest",
        )
    new_mask = new_cell_masks[node_index]
    if np.any(new_mask):
        if fill_new:
            ax.imshow(
                np.ma.masked_where(~new_mask, new_mask), cmap=new_cell_cmap,
                vmin=0, vmax=1, alpha=0.35, origin="upper", extent=image_extent,
                interpolation="nearest",
            )
        ax.contour(
            new_mask, levels=[0.5], colors=[NEW_CELL_COLOR],
            linewidths=0.6 if fill_new else 0.35,
            origin="upper", extent=image_extent,
        )
    ax.set_xlim(0, LX)
    ax.set_ylim(LY, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])



# Centered momentum components share the source grid and interval midpoints.
# D[0] is density, D[1] is omega_y, and D[2] is omega_x.
momentum_fields = {}
momentum_max = 0.0
if not method_data:
    raise ValueError(f"No retained methods for delta={DELTA:g}.")
for name, (solution, _, _, _, _) in method_data.items():
    momentum_y = np.asarray(solution.V.D[1])
    momentum_x = np.asarray(solution.V.D[2])
    if any(
        field.shape != (T_STEPS, NY, NX) or not np.all(np.isfinite(field))
        for field in (momentum_y, momentum_x)
    ):
        raise ValueError(f"Invalid centered momentum for {name}, delta={DELTA:g}.")
    magnitude = np.hypot(momentum_x, momentum_y)
    momentum_max = max(momentum_max, float(magnitude.max()))
    momentum_fields[name] = (momentum_x, momentum_y, magnitude)

# One length scale for all arrows; the largest vector spans two sampled grid
# spacings by default. Log lengths make weak fields visible alongside large peaks.
momentum_limit = momentum_max if momentum_max > 0 else 1.0
quiver_scale = (
    momentum_limit / (2 * QUIVER_STRIDE * max(DX, DY))
    if QUIVER_SCALE is None else float(QUIVER_SCALE)
)
if not np.isfinite(quiver_scale) or quiver_scale <= 0:
    raise ValueError("QUIVER_SCALE must be a finite positive number or None.")
momentum_norm = SymLogNorm(
    linthresh=max(0.001 * momentum_limit, np.finfo(float).tiny),
    vmin=0, vmax=momentum_limit, base=10,
)
momentum_cmap = plt.get_cmap("plasma")
quiver_sample = (slice(None, None, QUIVER_STRIDE),) * 2


def draw_momentum_overlay(ax, momentum_x, momentum_y, magnitude):
    sampled_magnitude = magnitude[quiver_sample]
    zero_mask = sampled_magnitude == 0
    arrow_x = momentum_x[quiver_sample]
    arrow_y = momentum_y[quiver_sample]
    if QUIVER_LOG_LENGTHS:
        display_magnitude = momentum_limit * np.asarray(momentum_norm(sampled_magnitude))
        length_factor = np.divide(
            display_magnitude, sampled_magnitude,
            out=np.zeros_like(sampled_magnitude, dtype=float), where=~zero_mask,
        )
        arrow_x = arrow_x * length_factor
        arrow_y = arrow_y * length_factor
    # angles='xy' respects the inverted image y-axis: positive omega_y points
    # down. Length mapping preserves direction; colors always use actual |omega|.
    return ax.quiver(
        X[quiver_sample], Y[quiver_sample],
        np.ma.masked_where(zero_mask, arrow_x),
        np.ma.masked_where(zero_mask, arrow_y),
        sampled_magnitude, cmap=momentum_cmap, norm=momentum_norm,
        angles="xy", scale_units="xy", scale=quiver_scale, pivot="mid",
        width=0.004, minlength=0, edgecolors="black", linewidths=0.2, zorder=3,
    )


# Three methods plus a repeated reference per row keep image panels readable.
method_names = list(method_data)
method_groups = [method_names[first:first + 3]
                 for first in range(0, len(method_names), 3)]
ncols = 1 + min(3, len(method_names))
side_margin, column_gap = 0.28, 0.10
panel_width = (FIGURE_WIDTH_IN - 2 * side_margin - (ncols - 1) * column_gap) / ncols
panel_height = panel_width * LY / LX
row_height = panel_height + 0.22
interval_height = len(method_groups) * row_height + 0.35
top_margin, bottom_margin = 0.32, 0.95
max_intervals = int((MAX_FIGURE_HEIGHT_IN - top_margin - bottom_margin) / interval_height)
if max_intervals < 1:
    raise ValueError("The image aspect ratio and method count do not fit the manuscript height.")
if OVERLAY_INTERVALS_PER_FIGURE > max_intervals:
    print(f"Using {max_intervals} intervals per PDF to fit the manuscript text height.")
    pages = interval_pages(T_STEPS, OVERLAY_INTERVALS, max_intervals)

overlay_figure_paths = []


def format_field_tick(value, position):
    """Keep shared colorbar labels readable at manuscript width."""
    if value == 0:
        return "$0$"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10.0**exponent
    return fr"${mantissa:.2g}\times10^{{{exponent}}}$"


with plt.rc_context({"font.size": 7, "axes.linewidth": 0.4, "pdf.fonttype": 42}):
    for page in pages:
        height = top_margin + len(page) * interval_height + bottom_margin
        fig = plt.figure(figsize=(FIGURE_WIDTH_IN, height))
        fig.text(0.5, 1 - 0.12 / height,
                 fr"Source and momentum | $\delta={DELTA:g}$",
                 ha="center", va="top", fontsize=8)
        cursor = height - top_margin
        for k in page:
            node_index = k + 1
            left, right = map(int, node_frames[k:k + 2])
            elapsed = (right - node_frames[0]) * FRAME_DT_MINUTES
            net_cells = int(cell_counts[node_index] - cell_counts[k])
            fig.text(side_margin / FIGURE_WIDTH_IN, cursor / height,
                     f"Frames ({left}, {right}] | {elapsed:g} min | "
                     f"{new_track_counts[node_index]} new tracks; {net_cells:+d} cells",
                     va="top", fontsize=7)
            cursor -= 0.25
            for group in method_groups:
                for col, name in enumerate([None, *group]):
                    x0 = side_margin + col * (panel_width + column_gap)
                    ax = fig.add_axes([x0 / FIGURE_WIDTH_IN,
                                       (cursor - row_height) / height,
                                       panel_width / FIGURE_WIDTH_IN,
                                       panel_height / height])
                    if name is None:
                        draw_microscope_overlay(ax, node_index, fill_new=True)
                        title = "New cells"
                    else:
                        draw_microscope_overlay(ax, node_index, source=source_fields[name][k])
                        mx, my, magnitude = momentum_fields[name]
                        draw_momentum_overlay(ax, mx[k], my[k], magnitude[k])
                        title = name.replace(" ACUOT", "\nACUOT").replace("-lineage", " lineage")
                    ax.set_title(title, fontsize=6.5, pad=2, linespacing=1.0)
                cursor -= row_height
            cursor -= 0.10

        # Independent legends distinguish signed source from momentum magnitude.
        bar_width = (FIGURE_WIDTH_IN - 2 * side_margin - 0.45) / 2
        for x0, norm, cmap, label, extend, ticks in [
            (side_margin, source_norm, source_cmap,
             r"$\zeta$ (cells / area / min)", source_colorbar_extend,
             [-source_limit, -source_norm.linthresh, 0, source_norm.linthresh, source_limit]),
            (side_margin + bar_width + 0.45, momentum_norm, momentum_cmap,
             r"$\|\omega\|$ (cells / length / min)", "neither",
             [0, momentum_norm.linthresh, momentum_limit]),
        ]:
            cax = fig.add_axes([x0 / FIGURE_WIDTH_IN, 0.70 / height,
                                bar_width / FIGURE_WIDTH_IN, 0.085 / height])
            colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                                    orientation="horizontal", extend=extend,
                                    ticks=np.unique(ticks),
                                    format=FuncFormatter(format_field_tick))
            colorbar.minorticks_off()
            colorbar.ax.tick_params(labelsize=5.5, length=2, pad=1)
            colorbar.set_label(label, fontsize=6, labelpad=2)
        fig.text(0.5, 0.12 / height,
                 "Cyan: new cells | Red/blue: source | Arrows: momentum\n"
                 f"Normalized spatial units; symlog colors; "
                 f"{'log' if QUIVER_LOG_LENGTHS else 'linear'} arrow lengths",
                 ha="center", va="bottom", fontsize=6, linespacing=1.3)
        if OVERLAY_PDF_DIR is not None:
            interval_tag = "-".join(f"{k + 1:02d}" for k in page)
            filename = (f"021-toiam-source-momentum_delta-{DELTA:g}_seq-{SEQUENCE}"
                        f"_frames-{node_frames[page[0]]}-{node_frames[page[-1] + 1]}"
                        f"_intervals-{interval_tag}.pdf")
            saved = save_manuscript_pdf(fig, Path(OVERLAY_PDF_DIR) / filename)
            overlay_figure_paths.append(saved)
            print(f"Saved {saved}")
        plt.show()
        plt.close(fig)
