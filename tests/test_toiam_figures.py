"""Checks for the TOIAM notebook's fixed penalty and publication exports."""

import contextlib
import io
import json
from pathlib import Path
import re
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.quiver import Quiver
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper"))
from toiam_figures import (
    DELTA, FIGURE_WIDTH_IN, MAX_FIGURE_HEIGHT_IN,
    interval_pages, save_manuscript_pdf,
)


class TestToiamFigures(unittest.TestCase):
    def test_interval_selection_and_last_partial_page(self):
        self.assertEqual(interval_pages(5, None, 2), [[0, 1], [2, 3], [4]])
        self.assertEqual(interval_pages(5, [4, 0, 2], 2), [[0, 2], [4]])
        for selected in ([], [0, 0], [-1], [5], [1.5], [True]):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                interval_pages(5, selected, 2)
        for size in (0, -1, 1.5, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                interval_pages(5, None, size)

    def test_pdf_geometry_even_with_global_tight_bbox(self):
        with tempfile.TemporaryDirectory() as directory:
            fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 2.4))
            ax.plot([0, 1], [0, 1])
            with plt.rc_context({"savefig.bbox": "tight"}):
                path = save_manuscript_pdf(fig, Path(directory) / "figure.pdf")
            box = re.search(rb"/MediaBox\s*\[([^]]+)\]", path.read_bytes())
            _, _, width, height = map(float, box.group(1).split())
            self.assertAlmostEqual(width, 370.38374 * 72 / 72.27, places=7)
            self.assertAlmostEqual(height, 2.4 * 72, places=7)
            fig.set_size_inches(FIGURE_WIDTH_IN, MAX_FIGURE_HEIGHT_IN + 0.1)
            with self.assertRaises(ValueError):
                save_manuscript_pdf(fig, path)
            plt.close(fig)

    def test_all_solver_calls_use_fixed_delta_and_reuse_endpoint_percentages(self):
        nb = json.loads((ROOT / "paper/008-toiam-dispersion-constraint.ipynb").read_text())
        for percentages, expected_calls in [([30, 50, 80], 6), ([0, 30, 100], 4)]:
            calls = []

            def solve(*args, **kwargs):
                calls.append(kwargs)
                return object(), object()

            ns = dict(np=np, DELTA=DELTA, computeGeodesic=solve,
                      observed_density=np.ones((3, 2, 2)), T_STEPS=2, LL=(2, 1, 1),
                      PPXA_ITERATIONS=100, PPXA_ALPHA=1.8, VERBOSE_SOLVER=False,
                      H_moments=[], GL_moments=[], GU_moments=[],
                      H_lineage=[], GL_full=[], GU_full=[],
                      configured_constraints={p: ([], [], []) for p in percentages})
            with contextlib.redirect_stdout(io.StringIO()):
                for i in (12, 13):
                    exec("".join(nb["cells"][i]["source"]), ns)
            self.assertEqual(len(calls), expected_calls)
            self.assertTrue(all(call["delta"] == 0.05 for call in calls))
            if 0 in percentages:
                self.assertIs(ns["configured_solutions"][0][0], ns["no_lineage_solution"])
            if 100 in percentages:
                self.assertIs(ns["configured_solutions"][100][0], ns["full_solution"])

    def test_source_and_momentum_share_panels_for_selected_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw, tracking = directory / "raw", directory / "tracking"
            raw.mkdir()
            tracking.mkdir()
            for frame in range(4):
                Image.fromarray(np.arange(192, dtype=np.uint8).reshape(16, 12)).save(raw / f"{frame}.tif")
                Image.fromarray(np.ones((16, 12), dtype=np.uint16)).save(tracking / f"{frame}.tif")
            fields = np.ones((3, 4, 3))
            methods = {}
            for m in range(6):
                v = SimpleNamespace(D=[fields, fields * m, fields * -m], Z=fields * m)
                methods[f"Method {m} ACUOT"] = (SimpleNamespace(V=v), None, None, None, None)
            y, x = np.meshgrid((np.arange(4) + 0.5) / 4,
                               (np.arange(3) + 0.5) / 4, indexing="ij")
            ns = dict(np=np, plt=plt, Image=Image, Path=Path, REPO_ROOT=ROOT,
                      TOIAM_ROOT=directory, SEQUENCE="test", RAW_IMAGE_DIR=raw,
                      env_path=lambda _: None,
                      index_tiffs=lambda p: {int(f.stem): f for f in Path(p).glob("*.tif")},
                      tracking_files={k: tracking / f"{k}.tif" for k in range(4)},
                      tracks={1: (0, 3, 0)}, node_frames=np.arange(4), image_shape=(16, 12),
                      T_STEPS=3, NY=4, NX=3, LX=0.75, LY=1, DX=0.25, DY=0.25, X=x, Y=y,
                      cell_counts=np.ones(4), FRAME_DT_MINUTES=1, method_data=methods,
                      RESULTS_DELTA=DELTA,
                      OVERLAY_INTERVALS=[2, 0], OVERLAY_INTERVALS_PER_FIGURE=99,
                      OVERLAY_PDF_DIR=directory / "pdfs")
            seen = []

            def inspect():
                fig = plt.gcf()
                self.assertLessEqual(fig.get_size_inches()[1], MAX_FIGURE_HEIGHT_IN)
                panels = [ax for ax in fig.axes if any(isinstance(c, Quiver) for c in ax.collections)]
                seen.extend(panels)
                for ax in panels:
                    self.assertEqual(len(ax.images), 2)  # microscope + source
                    self.assertGreater(ax.get_ylim()[0], ax.get_ylim()[1])

            with patch.object(plt, "show", inspect), contextlib.redirect_stdout(io.StringIO()):
                result = runpy.run_path(str(ROOT / "paper/toiam_momentum_overlay.py"), init_globals=ns)
            self.assertEqual(len(seen), 12)  # all six methods at both selected intervals
            names = [p.name for p in result["overlay_figure_paths"]]
            self.assertEqual(len(names), len(set(names)))
            self.assertTrue(all("delta-0.05" in name for name in names))
            self.assertEqual([k for page in result["pages"] for k in page], [0, 2])
            self.assertEqual(result["source_limit"], 5)
            self.assertAlmostEqual(result["momentum_limit"], np.sqrt(50))


if __name__ == "__main__":
    unittest.main()
