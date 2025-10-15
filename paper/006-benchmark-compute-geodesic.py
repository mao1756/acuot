"""
The code to measure the average computation time of geodesic calculation for each setting in the paper.
"""

# !/usr/bin/env python3
# benchmark_compute_geodesic.py

import os
import gc
import time
import math
import numpy as np
import scipy as sp
from scipy import stats
import sys

sys.path.append("../")
import proximal.dynamicUOT as dyn

# Base directory for test inputs (images/npz/npy). Override via env var if needed.
TEST_CASES_DIR = os.environ.get("TEST_CASES_DIR", "tests/proximal/test_cases")


# -----------------------------
# Scenario setup functions
# Each returns: (args_tuple, kwargs_dict) to pass into dyn.computeGeodesic
# -----------------------------


def setup_numpy_shk():
    # Define the initial and the terminal distributions
    T = 15
    K = 256

    X = np.linspace(0, 1, K)  # Discretization of the time-space domain
    rho_0 = np.exp(-0.5 * (X - 0.5) ** 2 / (0.05**2))
    rho_1 = 0.25 * np.exp(-0.5 * (X - 0.15) ** 2 / (0.05**2)) + 0.75 * np.exp(
        -0.5 * (X - 0.85) ** 2 / (0.05**2)
    )
    rho_0 /= np.sum(rho_0) / 256
    rho_1 /= np.sum(rho_1) / 256

    H = [[np.ones((T, K)), np.zeros((T, K)), np.zeros((T, K))]]
    GL = [np.ones(T)]
    GU = [np.ones(T)]
    args = (rho_0, rho_1, T, (1.0, 1.0))
    kwargs = dict(H=H, GL=GL, GU=GU, niter=10000)
    return args, kwargs


def setup_total_mass_inequality():
    def gauss(x, x_0, sigma, mass):
        normalized_factor = np.exp(-((x - x_0) ** 2) / sigma**2)
        return mass * (normalized_factor * K / np.sum(normalized_factor))

    sigma = 0.05
    K = 256
    X = np.linspace(0, 1, K)

    rho_0 = gauss(X, 0.25, sigma, 1)  # Initial density
    rho_1 = gauss(X, 0.75, sigma, 1)  # Final density

    rho_0 /= np.sum(rho_0) / 256
    rho_1 /= np.sum(rho_1) / 256

    T = 15
    ll = (1.0, 1.0)
    delta = 0.5 / np.pi

    H = [[np.ones((T, K)), np.zeros((T, K)), np.zeros((T, K))]]
    GL = [0.8 * np.ones((T,))]
    GU = [np.inf * np.ones((T,))]
    args = (rho_0, rho_1, T, ll)
    kwargs = dict(H=H, GL=GL, GU=GU, delta=delta, niter=3000)
    return args, kwargs


def setup_2d_total_mass():
    K = 30
    T = 15
    delta = 1.0
    rho_0 = np.load(os.path.join(TEST_CASES_DIR, "2D-total-mass-rho0.npy"))
    rho_1 = np.load(os.path.join(TEST_CASES_DIR, "2D-total-mass-rho1.npy"))

    t = np.array([(i + 0.5) / T for i in range(T)])
    F = 3 - 8 * (t - 0.5) ** 2
    GL = [F]
    GU = [F]
    H = [
        [
            np.ones((T, K, K)),
            np.zeros((T, K, K)),
            np.zeros((T, K, K)),
            np.zeros((T, K, K)),
        ]
    ]

    args = (rho_0, rho_1, T, (1.0, 1.0, 1.0))
    kwargs = dict(H=H, GL=GL, GU=GU, niter=3000, delta=delta)
    return args, kwargs


def _image_to_numpy(image_path):
    from PIL import Image

    image = Image.open(image_path).convert("L")
    return np.array(image)


def setup_barrier_static():
    maze = 1 - _image_to_numpy(os.path.join(TEST_CASES_DIR, "maze.png")).squeeze() / 255
    T = 30
    N1, N2 = maze.shape

    H = [
        [
            np.repeat(maze[np.newaxis, :, :], T, axis=0),
            np.zeros((T, N1, N2)),
            np.zeros((T, N1, N2)),
            np.zeros((T, N1, N2)),
        ]
    ]
    GL = [np.zeros(T)]
    GU = [np.zeros(T)]

    indices = np.arange(0, 30) * 1.0 / 30
    xx, yy = np.meshgrid(indices, indices)

    rho_0 = sp.stats.multivariate_normal.pdf(
        np.stack([xx, yy], axis=-1), mean=[5.0 / 30.0, 5.0 / 30.0], cov=2.0 / 36**2
    )
    rho_1 = sp.stats.multivariate_normal.pdf(
        np.stack([xx, yy], axis=-1), mean=[24.0 / 30.0, 24.0 / 30.0], cov=2.0 / 36**2
    )
    ll = (1.0, 1.0, 1.0)
    args = (rho_0, rho_1, T, ll)
    kwargs = dict(H=H, GL=GL, GU=GU, delta=10.0, niter=7000)
    return args, kwargs


def setup_barrier_moving():
    def fill_region(frames, fps, speed, original, Hstep):
        rows, cols = 30, 30
        step_size = speed * 14 / fps
        start_col, end_col = 1, 14
        filled_frames = [original]
        for i in range(1, frames):
            frame = filled_frames[-1].copy()
            end_fill = max(1, int(end_col - i * step_size))
            frame[18:20, end_fill:14] += Hstep
            frame[frame > 1] = 1
            filled_frames.append(frame.copy())
        return filled_frames

    maze = 1 - _image_to_numpy(os.path.join(TEST_CASES_DIR, "maze.png")).squeeze() / 255
    T = 30
    N1, N2 = maze.shape
    frames = fill_region(T, 30, 1.0, maze, Hstep=1.0)
    barrier = np.stack(frames, axis=0)

    H = [
        [
            barrier,
            np.zeros((T, N1, N2)),
            np.zeros((T, N1, N2)),
            np.zeros((T, N1, N2)),
        ]
    ]
    GL = [np.zeros(T)]
    GU = [np.zeros(T)]

    indices = np.arange(0, 30) * 1.0 / 30
    xx, yy = np.meshgrid(indices, indices)

    rho_0 = sp.stats.multivariate_normal.pdf(
        np.stack([xx, yy], axis=-1), mean=[5.0 / 30.0, 5.0 / 30.0], cov=2.0 / 36**2
    )
    rho_1 = sp.stats.multivariate_normal.pdf(
        np.stack([xx, yy], axis=-1), mean=[24.0 / 30.0, 24.0 / 30.0], cov=2.0 / 36**2
    )
    ll = (1.0, 1.0, 1.0)
    args = (rho_0, rho_1, T, ll)
    kwargs = dict(H=H, GL=GL, GU=GU, delta=10.0, niter=7000)
    return args, kwargs


def setup_ain():
    inputs = np.load(os.path.join(TEST_CASES_DIR, "ain-inputs.npz"))
    rho_0 = inputs["rho_0"]
    rho_1 = inputs["rho_1"]
    H1 = inputs["H1"]
    H2 = inputs["H2"]
    F1 = inputs["F1"]
    F2 = inputs["F2"]
    T = int(inputs["T"].item())
    ll = tuple(inputs["ll"])
    niter = int(inputs["niter"].item())
    N = rho_0.shape[0]
    H = [
        [H1, np.zeros((T, N, N)), np.zeros((T, N, N)), np.zeros((T, N, N))],
        [H2, np.zeros((T, N, N)), np.zeros((T, N, N)), np.zeros((T, N, N))],
    ]
    GL = [F1, F2]
    GU = [F1, F2]
    args = (rho_0, rho_1, T, ll)
    kwargs = dict(H=H, GL=GL, GU=GU, niter=niter, delta=1.0)
    return args, kwargs


def setup_curve_symmetric():
    def wrap(x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    def gaussian_on_circle(t, mu, sigma):
        return np.exp(-0.5 * wrap(t - mu) ** 2 / sigma**2)

    def rho_two_bumps(theta, mu=0.0, sigma=0.25):
        g1 = gaussian_on_circle(theta, mu, sigma)
        g2 = gaussian_on_circle(theta, mu + np.pi, sigma)
        return g1 + g2

    def normalize_density(rho, desired_mass=1.0):
        dx = 2 * np.pi / rho.size
        return rho / (rho.sum() * dx) * desired_mass

    def make_HF(theta, T):
        H = [
            np.tile(np.cos(theta), (T, 1)),
            np.tile(np.sin(theta), (T, 1)),
        ]
        F = [np.zeros(T), np.zeros(T)]
        return H, F

    sigma = 0.2
    T = 15
    K = 256
    ll = (1.0, 2 * np.pi)
    theta_grid = np.linspace(0.0, 2 * np.pi, K, endpoint=False)

    rho_0 = rho_two_bumps(theta_grid, mu=0.0, sigma=sigma)
    rho_1 = (
        gaussian_on_circle(theta_grid, mu=np.pi / 4, sigma=sigma)
        + gaussian_on_circle(theta_grid, mu=3 * np.pi / 4, sigma=sigma)
        + gaussian_on_circle(theta_grid, mu=5 * np.pi / 4, sigma=sigma)
        + gaussian_on_circle(theta_grid, mu=7 * np.pi / 4, sigma=sigma)
    )

    rho_0 = normalize_density(rho_0, desired_mass=1.0)
    rho_1 = normalize_density(rho_1, desired_mass=2.0)
    H, F = make_HF(theta_grid, T)
    H = [
        [H[0], np.zeros((T, K)), np.zeros((T, K))],
        [H[1], np.zeros((T, K)), np.zeros((T, K))],
    ]
    GL = F
    GU = F
    args = (rho_0, rho_1, T, ll)
    kwargs = dict(H=H, GL=GL, GU=GU, niter=10000, delta=0.01, periodic=True)
    return args, kwargs


def setup_curve_unsymmetric():
    def wrap(x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    def gaussian_on_circle(t, mu, sigma):
        return np.exp(-0.5 * wrap(t - mu) ** 2 / sigma**2)

    def rho_two_bumps(theta, mu=0.0, sigma=0.25):
        g1 = gaussian_on_circle(theta, mu, sigma)
        g2 = gaussian_on_circle(theta, mu + np.pi, sigma)
        return g1 + g2

    def normalize_density(rho, desired_mass=1.0):
        dx = 2 * np.pi / rho.size
        return rho / (rho.sum() * dx) * desired_mass

    def make_HF(theta, T):
        H = [
            np.tile(np.cos(theta), (T, 1)),
            np.tile(np.sin(theta), (T, 1)),
        ]
        F = [np.zeros(T), np.zeros(T)]
        return H, F

    sigma = 0.2
    T = 15
    K = 256
    ll = (1.0, 2 * np.pi)
    theta_grid = np.linspace(0.0, 2 * np.pi, K, endpoint=False)

    rho_0 = rho_two_bumps(theta_grid, mu=0.0, sigma=sigma)
    rho_1 = (
        gaussian_on_circle(theta_grid, mu=0, sigma=sigma)
        + gaussian_on_circle(theta_grid, mu=2 * np.pi / 3, sigma=sigma)
        + gaussian_on_circle(theta_grid, mu=4 * np.pi / 3, sigma=sigma)
    )

    rho_0 = normalize_density(rho_0, desired_mass=1.0)
    rho_1 = normalize_density(rho_1, desired_mass=2.0)
    H, F = make_HF(theta_grid, T)
    H = [
        [H[0], np.zeros((T, K)), np.zeros((T, K))],
        [H[1], np.zeros((T, K)), np.zeros((T, K))],
    ]
    GL = F
    GU = F
    args = (rho_0, rho_1, T, ll)
    kwargs = dict(H=H, GL=GL, GU=GU, niter=10000, delta=0.01, periodic=True)
    return args, kwargs


def setup_river():
    maze = np.zeros((30, 30))
    maze[13:17, 0:30] = 1

    T = 15
    N1, N2 = maze.shape

    H1 = np.repeat(maze[np.newaxis, :, :], T, axis=0) * np.cos(np.pi / 4)
    H2 = np.repeat(maze[np.newaxis, :, :], T, axis=0) * np.sin(np.pi / 4)

    indices = np.arange(0, 30) * 1.0 / 30
    xx, yy = np.meshgrid(indices, indices)

    rho_0 = sp.stats.multivariate_normal.pdf(
        np.stack([xx, yy], axis=-1), mean=[15.0 / 30.0, 20.0 / 30.0], cov=2.0 / 36**2
    )
    rho_1 = sp.stats.multivariate_normal.pdf(
        np.stack([xx, yy], axis=-1), mean=[15.0 / 30.0, 8.0 / 30.0], cov=2.0 / 36**2
    )

    ll = (1.0, 1.0, 1.0)
    delta = 2.0
    Hs = [[np.zeros((T, N1, N2)), H1, H2, np.zeros((T, N1, N2))]]
    GL = [np.zeros((T,))]
    GU = [np.ones((T,)) * np.inf]
    args = (rho_0, rho_1, T, ll)
    kwargs = dict(H=Hs, GL=GL, GU=GU, p=2.0, q=2.0, delta=delta, niter=3000)
    return args, kwargs


def setup_budget():
    def gauss(x, x_0, sigma, mass, K):
        normalized_factor = np.exp(-((x - x_0) ** 2) / sigma**2)
        return mass * (normalized_factor * K / np.sum(normalized_factor))

    sigma = 0.03
    K = 256
    X = np.linspace(0, 1, K)

    rho_0 = gauss(X, 0.25, sigma, 1, K)
    rho_1 = gauss(X, 0.75, sigma, 1, K)

    rho_0 /= np.sum(rho_0) / 256
    rho_1 /= np.sum(rho_1) / (256 * 2)

    rho_0 = rho_0.astype(np.float64)
    rho_1 = rho_1.astype(np.float64)

    T = 15
    ll = (1.0, 1.0)

    # Step mask boundary (right half)
    H_start_x = 0.5
    j0 = int(np.clip(int(np.floor(H_start_x * K)), 0, K - 1))

    H_vec = np.zeros((K,), dtype=float)
    H_vec[j0:] = 1.0
    H = np.tile(H_vec, (T, 1))

    Hs = [[np.zeros((T, K)), np.zeros((T, K)), H]]
    GL = [np.zeros((T,))]
    GU = [0.1 * np.ones((T,))]
    args = (rho_0, rho_1, T, ll)
    kwargs = dict(H=Hs, GL=GL, GU=GU, p=2.0, q=2.0, delta=0.5 / np.pi, niter=3000)
    return args, kwargs


# -----------------------------
# Timing harness
# -----------------------------


def time_compute_geodesic(setup_fn, tries=3):
    """
    For a given setup function, run dyn.computeGeodesic 3 times and return list of durations (seconds).
    Only the computeGeodesic call is timed; setup work is excluded.
    """
    times = []
    last_result = None  # keep a reference to avoid premature GC during timing
    for i in range(tries):
        # Build fresh inputs (not timed)
        args, kwargs = setup_fn()
        gc.collect()
        t0 = time.perf_counter()
        last_result = dyn.computeGeodesic(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        # Drop heavy results between runs to minimize cross-run interference
        del last_result
        gc.collect()
    return times


def main():
    benchmarks = {
        "numpy_shk": setup_numpy_shk,
        "total_mass_inequality": setup_total_mass_inequality,
        "2d_total_mass": setup_2d_total_mass,
        "barrier_static": setup_barrier_static,
        "barrier_moving": setup_barrier_moving,
        "ain": setup_ain,
        "curve_symmetric": setup_curve_symmetric,
        "curve_unsymmetric": setup_curve_unsymmetric,
        "river": setup_river,
        "budget": setup_budget,
    }

    print("Benchmarking dyn.computeGeodesic (3 tries per function)\n")
    header = f"{'Function':<22} {'Mean (s)':>12} {'Std (s)':>12}  Runs (s)"
    print(header)
    print("-" * len(header))

    for name, setup_fn in benchmarks.items():
        try:
            durations = time_compute_geodesic(setup_fn, tries=3)
            mean = float(np.mean(durations))
            std = float(np.std(durations, ddof=1)) if len(durations) > 1 else 0.0
            runs_str = ", ".join(f"{d:.3f}" for d in durations)
            print(f"{name:<22} {mean:>12.3f} {std:>12.3f}  {runs_str}")
        except FileNotFoundError as e:
            print(f"{name:<22} {'N/A':>12} {'N/A':>12}  skipped (missing file: {e})")
        except Exception as e:
            print(f"{name:<22} {'ERR':>12} {'ERR':>12}  error: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
