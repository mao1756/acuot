"""Algorithms for dynamic unbalanced optimal transport solvers.

This module collects the proximal operators, Poisson solvers, projection
utilities, and Douglas-Rachford/PPXA schemes that power the dynamic
unbalanced optimal transport (UOT) routines in :mod:`proximal`.  The
implementations operate on the grid objects defined in :mod:`proximal.grids`
and are backend agnostic thanks to the :mod:`ot.backend` abstraction, so they
work with both NumPy- and torch-based pipelines.
"""

import proximal.grids as grids
from proximal.backend_extension import get_backend_ext
from proximal.backend_extension import NumpyBackend_ext, TorchBackend_ext
from ot.backend import Backend
from ot.utils import list_to_array
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import math
from typing import List


def root(a, b, c, d, nx: Backend):
    """Return the largest real root of the cubic ``a x^3 + b x^2 + c x + d``.

    All coefficients are broadcast elementwise, so vector inputs yield the
    elementwise largest real root.  Complex intermediate values are handled via
    the ``nx`` backend, allowing NumPy- or torch-based execution.

    Args:
        a (array-like): Coefficient multiplying ``x**3`` (must be non-zero).
        b (array-like): Coefficient multiplying ``x**2``.
        c (array-like): Coefficient multiplying ``x``.
        d (array-like): Constant term.
        nx (Backend): Numerical backend (e.g. NumPy or torch) that supplies the
            primitive operations.

    Returns:
        array-like: Largest real root for each entry in the broadcast inputs.
    """
    assert (
        a.shape == b.shape == c.shape == d.shape
    ), "Coefficients must have the same shape"
    z = nx.zeros(a.shape, type_as=a)
    u = nx.zeros(a.shape, type_as=a)
    v = nx.zeros(a.shape, type_as=a)
    # Transform coefficients
    p = -((b / a) ** 2) / 3 + c / a
    q = 2 * (b / a) ** 3 / 27 - (b * c) / (a**2) / 3 + d / a
    delta = q**2 + 4 / 27 * p**3
    id = delta > 0
    u[id] = nx.cbrt((-q[id] + nx.sqrt(delta[id])) / 2)
    v[id] = nx.cbrt((-q[id] - nx.sqrt(delta[id])) / 2)
    z[id] = u[id] + v[id] - b[id] / a[id] / 3

    id = delta < 0
    u[id] = (((-q[id] + 1j * nx.sqrt(-delta[id])) / 2) ** (1 / 3)).real
    z[id] = 2 * u[id] - b[id] / a[id] / 3

    id = delta == 0
    z[id] = nx.where(
        q[id] == 0,
        -b[id] / a[id] / 3,
        3 * q[id] / p[id] - b[id] / a[id] / 3,
    )

    return z


def mdl(M, nx: Backend):
    """Return the pointwise Euclidean norm of a list of aligned arrays."""
    return nx.sqrt(sum(m**2 for m in M))


def proxA_(dest: grids.Cvar, M, gamma: float):
    """Project a stacked flux ``M`` onto the scaled L2 ball.

    The routine applies the proximal operator of ``gamma * ||M||_2`` by applying a
    soft-thresholding factor to each component and writing the result into
    ``dest`` in place.

    Args:
        dest (grids.Cvar): Destination container receiving the projected fluxes.
        M (Sequence[array-like]): Sequence of flux components with identical
            shapes.
        gamma (float): Step size for the proximal operator.
    """
    softth = dest.nx.maximum(1 - gamma / mdl(M, dest.nx), 0.0)
    for k in range(len(M)):
        dest[k] = softth * M[k]


def proxB_(destR, destM: list, R, M: list, gamma: float, nx: Backend):
    r"""Compute the proximal map of ``sum |M_i|^2 / R`` under positivity constraints.

    The update solves

    .. math::
        \operatorname{prox}_{\gamma}(R, M)
            = \arg\min_{\tilde R, \tilde M}
              \tfrac{1}{2}\|\tilde R - R\|_2^2 + \tfrac{1}{2}\sum_i\|\tilde M_i - M_i\|_2^2
              + \gamma \sum_i \frac{\|\tilde M_i\|_2^2}{\tilde R}

    subject to ``\tilde R >= 0``.  The result is written into ``destR`` and
    ``destM`` in place.

    Args:
        destR (array-like): Output buffer for the updated mass term ``R``.
        destM (Sequence[array-like]): Output buffers for the associated fluxes.
        R (array-like): Current mass iterate.
        M (Sequence[array-like]): Current flux iterates.
        gamma (float): Proximal step size.
        nx (Backend): Numerical backend providing linear-algebra primitives.
    """
    a = nx.ones(R.shape, type_as=R)
    b = 2 * gamma - R
    c = gamma**2 - 2 * gamma * R
    d = -(gamma / 2) * mdl(M, nx) ** 2 - gamma**2 * R
    destR[...] = nx.maximum(0.0, root(a, b, c, d, nx))
    DD = nx.zeros(R.shape, type_as=R)
    DD[destR > 0] = 1.0 - gamma / (gamma + destR[destR > 0])
    for k in range(len(M)):
        destM[k][...] = DD * M[k]


def proxF_(dest: grids.Cvar, V: grids.Cvar, gamma: float, p: float, q: float):
    """Evaluate the proximal map of the energy functional ``F``.

    The centred-grid variable ``V`` stores density and momentum terms whose
    proximal update depends on the choice of ``(p, q)`` combination.  The helper
    dispatches to the appropriate proximal operator (``proxA_``/``proxB_``)
    depending on whether the problem corresponds to classical Wasserstein (W1,
    W2), Wasserstein-Fisher-Rao (WFR), or their mixed variants.

    Args:
        dest (grids.Cvar): Output container updated in place.
        V (grids.Cvar): Input centred-grid variable.
        gamma (float): Proximal step size.
        p (float): Lp exponent that selects the transport norm (1 or 2).
        q (float): Lq exponent controlling the source term (<= 2).
    """
    if p == 1 and q < 1:  # W1
        dest.D[0][:] = V.D[0]
        proxA_(dest.D[1:], V.D[1:], gamma)
    elif p == 2 and q < 1:  # W2
        proxB_(dest.D[0], dest.D[1:], V.D[0], V.D[1:], gamma, dest.nx)
    elif p == 1 and q == 1:  # "Bounded Lipschitz"
        dest.D[0][:] = V.D[0]
        proxA_(dest.D[1:], V.D[1:], gamma)
        proxA_(dest.Z, V.Z, gamma)
    elif p == 2 and q == 2:  # WFR
        proxB_(
            dest.D[0],
            dest.D[1:] + [dest.Z],
            V.D[0],
            V.D[1:] + [V.Z],
            gamma,
            dest.nx,
        )
    elif p == 2 and q == 1:  # Partial W2
        proxB_(dest.D[0], dest.D[1:], V.D[0], V.D[1:], gamma, dest.nx)
        proxA_(dest.Z, V.Z, gamma)
    elif p == 1 and q == 2:  # W1-FR
        proxA_(dest.D[1:], V.D[1:], gamma)
        proxB_(dest.D[0], dest.Z, V.D[0], V.Z, gamma, dest.nx)
    else:
        raise ValueError("Functional not implemented")


def poisson_(f, ll, source, nx: Backend):
    """Solve a Poisson problem or its screened variant with Neumann boundary data.

    The centred-grid array ``f`` is overwritten with the solution of

    - ``Delta u + f = 0`` when ``source`` is ``False``
    - ``(Delta - 1) u + f = 0`` when ``source`` is ``True``

    using separable discrete cosine transforms (DCT-II) along each axis.

    Args:
        f (array-like): Right-hand side on the centred grid (modified in place).
        ll (tuple[float, ...]): Physical lengths of each domain axis.
        source (bool): Whether to include the ``-u`` source term.
        nx (Backend): Numerical backend providing DCT and linear-algebra ops.
    """
    d = f.ndim
    N = f.shape
    h = [length / n for length, n in zip(ll, N)]
    dims = [1] * d
    D = nx.zeros(f.shape, type_as=f)

    for k in range(d):
        dims = [1] * d
        dims[k] = N[k]
        dep = nx.zeros(tuple(dims), type_as=f)
        for i in range(N[k]):
            slices = [slice(None)] * d
            slices[k] = i
            slices = tuple(slices)
            dep[slices] = (2 * math.cos(math.pi * i / N[k]) - 2) / h[k] ** 2
        D += dep

    if source:
        D -= 1
    else:
        D[0] = 1

    # axis wise DCT
    for axe in range(d):
        f[...] = nx.dct(f, axis=axe, norm="ortho")
    f /= -D
    for axe in range(d):
        f[...] = nx.idct(f, axis=axe, norm="ortho")


def poisson_mixed_bc_(f, ll, source, nx, periodic_axes=None):
    """Solve a Poisson problem with mixed Neumann/periodic boundary conditions.

    Axis 0 is treated with Neumann conditions using a DCT-II transform, while
    the axes listed in ``periodic_axes`` use FFTs.  The routine overwrites
    ``f`` with the solution of ``Delta u + f = 0`` (or ``(Delta - 1)u + f = 0`` when
    ``source`` is ``True``).

    Args:
        f (array-like): Right-hand side on the centred grid (modified in place).
        ll (tuple[float, ...]): Physical lengths of each axis.
        source (bool): Whether to include the ``-u`` term (Helmholtz case).
        nx (Backend): Numerical backend providing FFT/DCT implementations.
        periodic_axes (Iterable[int], optional): Indices of periodic axes; any
            axis not listed is treated with Neumann boundary conditions.
    """
    d = f.ndim
    if periodic_axes is None:
        periodic_axes = tuple(range(1, d))  # all spatial axes
    N = f.shape
    h = [L / n for L, n in zip(ll, N)]

    # --- build the diagonal Lambda(k0,...,kd-1) -----------------------
    D = nx.zeros(f.shape, type_as=f)
    for ax in range(d):
        k = nx.arange(N[ax], type_as=f)  # mode indices
        if ax in periodic_axes:  # periodic
            lam = (2 * nx.cos(2 * math.pi * k / N[ax]) - 2) / h[ax] ** 2
        else:  # Neumann
            lam = (2 * nx.cos(math.pi * k / N[ax]) - 2) / h[ax] ** 2
        # broadcast along the other axes
        shape = [1] * d
        shape[ax] = N[ax]
        D += lam.reshape(shape)

    if source:  # Solve (Delta - 1)u = -f
        D -= 1.0
    else:  # pure Poisson: make Lambda(0,...,0)=1 to avoid 0-div
        D[(0,) * d] = 1.0

    # --- forward transforms -----------------------------------------
    # time axis (Neumann)
    fhat = nx.dct(f, axis=0, norm="ortho")

    # spatial axes (periodic)
    for ax in periodic_axes:
        fhat = nx.fft(fhat, axis=ax, norm="ortho")  # complex output
        # We note that writing f[...] = nx.fft(fhat, axis=ax, norm="ortho") would throw away the imaginary part since the
        # f is real-valued, so we need to keep the complex output in fhat.
    # --- diagonal solve ---------------------------------------------
    fhat /= -D  # broadcast div
    # --- inverse transforms (reverse order) -------------------------
    for ax in reversed(periodic_axes):
        fhat = nx.ifft(fhat, axis=ax, norm="ortho").real

    f[...] = nx.idct(fhat, axis=0, norm="ortho")  # back to space


def minus_interior_(dest, M, dpk, cs, dim):
    """Overwrite the interior slice of ``dest`` with ``M - dpk``.

    All arrays follow the staggered-grid convention used by
    :class:`grids.Svar`, where axis ``dim`` stores an extra cell.  The update is
    restricted to the interior cells - the entries that exclude the first element
    along ``dim``.

    Args:
        dest (array-like): Output array with the same shape as ``M``.
        M (array-like): Staggered-grid array containing the current values.
        dpk (array-like): Difference array defined on the centred grid.
        cs (tuple[int, ...]): Shape of the centred grid used for slicing.
        dim (int): Axis along which the variable is staggered.
    """
    assert dest.shape == M.shape, "Destination and source shapes must match"
    slices = [slice(None)] * M.ndim
    slices[dim] = slice(1, cs[dim])

    interior_diff = M[tuple(slices)] - dpk
    dest[tuple(slices)] = interior_diff


def projCE_(
    dest: grids.Svar, U: grids.Svar, rho_0, rho_1, source: bool, periodic: bool = False
):
    """Project a staggered variable onto the continuity equation subspace.

    The projection enforces ``d_t rho + div(omega) = zeta`` (or its
    periodic variant) by solving a Poisson problem for the pressure potential
    and subtracting its discrete gradient from ``U``.  Results are written into
    ``dest`` in place.

    Args:
        dest (grids.Svar): Output variable storing the projected staggered field.
        U (grids.Svar): Input staggered variable before projection.
        rho_0 (array-like): Source density (initial condition).
        rho_1 (array-like): Target density (terminal condition).
        source (bool): Whether the formulation includes source terms (``q >= 1``).
        periodic (bool, optional): Use periodic boundary conditions in space.  The
            time axis always keeps Neumann conditions.
    """

    assert dest.ll == U.ll, "Destination and source lengths must match"

    U.proj_BC(rho_0, rho_1, periodic=periodic)
    p = -U.remainder_CE(periodic=periodic)  # p = Z - div(U)
    if periodic:
        poisson_mixed_bc_(p, U.ll, source, dest.nx)
    else:
        poisson_(p, U.ll, source, dest.nx)

    for k in range(len(U.cs)):
        # In periodic case, subtract the interior for the time dim.
        # Otherwise (space dims), subtract entirely (because there is no boundary in the first place)
        # Note that the gradient for the forward divergence is the backward difference
        if periodic and k != 0:
            dpk = (p - dest.nx.roll(p, 1, axis=k)) * U.cs[k] / U.ll[k]
            dest.D[k][...] = U.D[k] - dpk
        else:
            dpk = dest.nx.diff(p, axis=k) * U.cs[k] / U.ll[k]

            minus_interior_(dest.D[k], U.D[k], dpk, U.cs, k)

    dest.proj_BC(rho_0, rho_1, periodic=periodic)
    if source:
        dest.Z[...] = U.Z - p


def invQ_mul_A_(dest, Q, src, dim: int, nx: Backend):
    """Apply ``Q^{-1}`` to each 1-D fibre of ``src`` along axis ``dim``.

    The helper reshapes the array so that slices along ``dim`` become columns,
    solves the linear systems using ``nx.solve``, and writes the result to
    ``dest``.  Shapes are preserved, and the operation is performed in place.

    Args:
        dest (array-like): Output buffer receiving the transformed values.
        Q (array-like): Square matrix defining the 1-D coupling to invert.
        src (array-like): Input data whose fibres are transformed.
        dim (int): Axis along which to apply the inverse.
        nx (Backend): Numerical backend used for solving the linear systems.
    """

    # Put the dimension to the first axis
    new_axes = (dim,) + tuple(i for i in range(src.ndim) if i != dim)
    one_d_slices = nx.transpose(src, axes=new_axes)
    dim_shape = one_d_slices.shape[0]
    remaining_shape = one_d_slices.shape[1:]

    # Reshape the result to 2D
    one_d_slices = one_d_slices.reshape(dim_shape, -1)

    # Put the batch dimension to the first axis
    # one_d_slices = nx.transpose(one_d_slices)

    # Apply the inverse of Q to each slice
    invQ_slices = nx.solve(Q, one_d_slices)

    # Reshape the result back to the original shape
    invQ_slices = invQ_slices.reshape(-1, *remaining_shape)

    # Put the dimension back to the original axis
    inverse_axes = tuple(0 if i == dim else i + 1 - (i >= dim) for i in range(src.ndim))
    invQ_slices = nx.transpose(invQ_slices, axes=inverse_axes)

    # Put the result back to the original array
    dest[...] = invQ_slices


def projinterp_(dest: grids.CSvar, x: grids.CSvar, Q, log=None):
    r"""Project ``(U, V)`` onto the linear constraint ``V = I(U)``.

    Given an input pair ``x = (U, V)``, the routine computes

    .. math:: U' = (\mathrm{Id} + I^\top I)^{-1}(U + I^\top V), \qquad
              V' = I(U')

    where each dimension has its own precomputed matrix ``Q[k] = Id + I^T I``.
    The projected pair ``(U', V')`` is written to ``dest`` in place.

    Args:
        dest (grids.CSvar): Destination variable storing the projected pair.
        x (grids.CSvar): Input centred/staggered variables ``(U, V)``.
        Q (Sequence[array-like]): Per-dimension matrices representing
            ``Id + I^T I`` on the centred grid.
        log (dict, optional): Diagnostics dictionary; when provided, the norm of
            ``U'`` is appended under the ``"U_prime_interp"`` key.
    """
    assert dest.ll == x.ll, "Length scales must match"

    x_U_copy = x.U.copy()  # Copy x.U since interpT_ will overwrite x.U if dest=x

    # Calculate I*V and store it in U'
    grids.interpT_(dest.U, x.V, dest.periodic)
    # Add U to I*V and store it in U'... (*)
    dest.U += x_U_copy

    # Apply inverse Q matrix operation for each dimension
    for k in range(dest.U.N):
        invQ_mul_A_(dest.U.D[k], Q[k], dest.U.D[k], k, dest.nx)

    # Average source terms (* adds x.V.Z(moved to dest.U by interpT_) and x.U.Z, so we
    # only need to divide by 2 here)
    dest.U.Z *= 0.5

    # Log the norm of U'
    if log is not None:
        dens_norm = 0
        for dens in dest.U.D:
            dens_norm += dest.nx.norm(dens) ** 2
        log["U_prime_interp"].append(dest.nx.sqrt(dens_norm))

    # Calculate V' = I(U) and store it in V'
    dest.interp_()


def projinterp_constraint_(dest: grids.CSvar, x: grids.CSvar, Q, HQH, H, F, log=None):
    r"""Project onto interpolation and affine constraints simultaneously.

    Given ``x = (U, V)``, the projection solves

    .. math::
        \lambda = (H (\mathrm{Id}+I^T I)^{-1} H^*)^{-1}
            (H (\mathrm{Id}+I^T I)^{-1}(U + I^T V) - F)

    followed by

    .. math::
        U' = (\mathrm{Id}+I^T I)^{-1}(U + I^T V - H^* \lambda), \qquad
        V' = I(U').

    The tensors ``Q`` and ``HQH`` store the matrices ``Id + I^T I`` and
    ``H (Id + I^T I)^{-1} H^*`` respectively, so that the projection can be
    assembled efficiently.  When ``log`` is provided, norms of the intermediate
    quantities are appended for diagnostics.

    Args:
        dest (grids.CSvar): Output variable receiving ``(U', V')``.
        x (grids.CSvar): Input pair prior to projection.
        Q (Sequence[array-like]): Per-dimension matrices for ``Id + I^T I``.
        HQH (array-like): Matrix ``H (Id + I^T I)^{-1} H^*`` used to recover the
            Lagrange multipliers.
        H (array-like): Discretised constraint operator ``H``.
        F (array-like): Right-hand side of the affine constraint.
        log (dict, optional): Diagnostics dictionary.  Expected keys include
            ``"pre_lambda"``, ``"lambda"``, ``"lambda_eqn"``, and
            ``"U_prime_interp"``.
    """
    if log is None:
        log = {
            "pre_lambda": [],
            "lambda": [],
            "first_order_condition": [],
            "U_prime_interp": [],
            "HQHlambda": [],
            "lambda_eqn": [],
        }
    projinterp_(dest, x, Q, log)  # Calculate U'=Q^{-1}(U+I*V) and V'=I(U)

    # Calculate HU'-F=H(Id+I^* I)^{-1}(U+I*V)-F
    pre_lambda = (
        dest.nx.sum(H * dest.V.D[0], axis=tuple(range(1, H.ndim)))
        * (math.prod(dest.ll[1:]) / math.prod(dest.cs[1:]))
        - F
    )
    log["pre_lambda"].append(dest.nx.norm(pre_lambda))

    # Calculate lambda = (H(Id+I^* I)^{-1}H*)^{-1}(HU'-F)
    lambda_ = dest.nx.solve(HQH, pre_lambda)
    log["lambda"].append(dest.nx.norm(lambda_))

    # Verify the equation HQH lambda = pre_lambda
    log["lambda_eqn"].append(dest.nx.norm(HQH @ lambda_ - pre_lambda))

    # Calculate h* lambda
    Hstar_lambda = (
        H
        * lambda_[(slice(None),) + (None,) * (H.ndim - 1)]
        * (math.prod(dest.ll[1:]) / math.prod(dest.cs[1:]))
    )
    # Hstar_lambda = I*(Hstar_lambda), apply the adjoint of the interpolation operator
    dk = list(Hstar_lambda.shape)
    dk[0] = 1
    dk = tuple(dk)
    slices = [slice(None)] * Hstar_lambda.ndim
    slices[0] = slice(0, Hstar_lambda.shape[0] + 1)
    cat = dest.nx.concatenate(
        [
            dest.nx.zeros(dk, type_as=Hstar_lambda),
            Hstar_lambda,
            dest.nx.zeros(dk, type_as=Hstar_lambda),
        ],
        axis=0,
    )
    Hstar_lambda = ((cat + dest.nx.roll(cat, -1, axis=0)) / 2)[tuple(slices)]
    # Copy for debugging
    # Hstar_lambda_copy = Hstar_lambda.copy()

    # Calculate U' = Q^{-1}H^* lambda
    invQ_mul_A_(Hstar_lambda, Q[0], Hstar_lambda, 0, dest.nx)

    # HQ^{-1}H^* lambda = HQHlambda. This should be equal to HQ^{-1}(U+I*V)
    HQHlambda = [dest.nx.zeros(d.shape) for d in dest.U.D]
    HQHlambda[0] = Hstar_lambda
    U_HQHlambda = grids.Svar(x.cs, x.ll, HQHlambda, dest.nx.zeros(x.cs))
    V_HQHlambda = grids.interp(U_HQHlambda)
    HV_HQHlambda = (
        dest.nx.sum(H * V_HQHlambda.D[0], axis=tuple(range(1, H.ndim)))
        * math.prod(dest.ll[1:])
        / math.prod(dest.cs[1:])
    )
    # Calculate the difference
    diff = pre_lambda - HV_HQHlambda
    log["HQHlambda"].append(dest.nx.norm(diff))

    # Calculate U' = Q^{-1}(U+I*V)-Q^{-1}H^* lambda
    dest.U.D[0] -= Hstar_lambda

    # Calculate V' = I(U)
    dest.interp_()

    # Checking if the first order condition
    # U'-U_copy + I*(V'-V_copy) + H^* lambda = 0
    # is satisfied

    # Calculate I*(V'-V_copy)
    V_diff = dest.V - x.V
    interpT_V_diff = grids.interpT(V_diff)

    # Calculate the staggered variable H^* lambda
    # U_hstar_lambda = grids.CSvar(x.rho0, x.rho1, x.cs[0], x.ll)
    # U_hstar_lambda.U.D[0] = Hstar_lambda_copy
    # U_hstar_lambda = U_hstar_lambda.U

    # Calculate U'-U_copy + I*(V'-V_copy) + H^* lambda
    # first_order_condition = dest.U - x.U + interpT_V_diff + U_hstar_lambda

    # dens_norm = 0
    # Calculate the norm for each dimension
    # for dens in first_order_condition.D:
    #    dens_norm += dest.nx.norm(dens) ** 2
    # log["first_order_condition"].append(dest.nx.sqrt(dens_norm))


def periodic_interpolation_matrix(N, nx: Backend, type_as=None):
    """Return the N-by-N matrix ``I_N`` with 0.5 on the main and +1 cyclic diagonals."""
    I_N = nx.zeros((N, N), type_as=type_as)
    idx = nx.arange(N)
    I_N[idx, idx] = 0.5  # main diagonal
    I_N[idx, (idx + 1) % N] = 0.5  # super-diagonal with wrap-around
    return I_N


def itit_plus_identity(N, nx: Backend, type_as=None):
    """Return ``I_N.T @ I_N + Id`` for the periodic interpolation matrix ``I_N``.

    The result is a circulant, tridiagonal-with-wrap matrix whose entries are
    1.5 on the diagonal and 0.25 on the +/- 1 cyclic off-diagonals.
    """
    idx = nx.arange(N)
    M = nx.zeros((N, N), type_as=type_as)
    M[idx, idx] = 1.5
    M[idx, (idx + 1) % N] = 0.25
    M[idx, (idx - 1) % N] = 0.25
    return M


def precomputeProjInterp(cs, rho0, rho1, periodic=False):
    B = []
    nx = get_backend_ext(rho0, rho1)
    for i in range(len(cs)):
        n = cs[i]
        if periodic and i != 0:
            # For the space dimension, use the periodic interpolation matrix
            Q = itit_plus_identity(n, nx, type_as=rho0)
            B.append(Q)
        else:
            # Create a tridiagonal matrix
            main_diag = nx.full((n + 1,), 6, type_as=rho0)
            main_diag[0], main_diag[-1] = 5, 5  # Adjust the first and last element
            off_diag = nx.ones(n, type_as=rho0)
            Q = nx.diag(off_diag, -1) + nx.diag(main_diag, 0) + nx.diag(off_diag, 1)
            Q /= 4
            # Store the result
            B.append(Q)
    return B


def precomputeHQH(Q, H, cs, ll):
    """Build ``H (Id + I^T I)^{-1} H^T`` for the interpolation constraint.

    Args:
        Q (array-like): Square block ``Id + I^T I`` associated with the temporal
            interpolation operator.
        H (array-like): Constraint kernel evaluated on the centred grid.
        cs (tuple[int, ...]): Grid resolution (used for scaling factors).
        ll (tuple[float, ...]): Physical lengths (used for integration weights).
    """
    nx = get_backend_ext(Q, H)
    H_sum = nx.sum(H**2, axis=tuple(range(1, H.ndim))).reshape(-1, H.shape[0])
    Q_inv = nx.inv(Q)
    Q_plus_Q = Q_inv[:, :-1] + Q_inv[:, 1:]
    IQ_plus_Q = ((Q_plus_Q + nx.roll(Q_plus_Q, -1, axis=0)) / 4)[:-1]
    return H_sum * IQ_plus_Q * (math.prod(ll[1:]) / math.prod(cs[1:])) ** 2


def flatten_array(arr, fastest_axis: int = 0):
    """
    Flatten `arr` into a 1D vector, making `fastest_axis` the one
    that changes most rapidly in the flattened ordering.

    Parameters
    ----------
    arr : array-like
        The input N-dimensional array.
    fastest_axis : int, default=0
        The index (axis) which should vary the fastest in the flattened vector.

    Returns
    -------
    flat : array-like (1D)
        The flattened array in the specified order.
    """
    # Get the backend module
    nx = get_backend_ext(arr)

    # Number of dimensions
    nd = arr.ndim

    # Create a list of axes [0, 1, 2, ..., nd-1]
    all_axes = list(range(nd))

    # Swap the `fastest_axis` with the last axis
    all_axes[fastest_axis], all_axes[-1] = all_axes[-1], all_axes[fastest_axis]

    # Transpose arr so that `fastest_axis` is last
    transposed = nx.transpose(arr, axes=all_axes)

    # Now flatten in standard (row-major) order
    flat = transposed.reshape(-1)

    return flat


def unflatten_array(vec, shape: tuple[int, ...], fastest_axis: int = 0):
    """
    Reshape the 1D array `vec` back to the original `shape` (N-dimensional),
    such that the axis `fastest_axis` becomes the same fastest axis
    used in flatten_array().

    Parameters
    ----------
    vec : array-like (1D)
        The flattened array.
    shape : tuple of ints
        The target shape of the N-dimensional array.
    fastest_axis : int, default=0
        The axis that was set to vary fastest in flatten_array().

    Returns
    -------
    arr : array-like (N-dimensional)
        The reshaped array matching `shape`.
    """
    nx = get_backend_ext(vec)
    nd = len(shape)

    # The same axis-reordering that we used in flatten_array
    all_axes = list(range(nd))
    all_axes[fastest_axis], all_axes[-1] = all_axes[-1], all_axes[fastest_axis]

    # We'll reshape `vec` into the transposed shape
    # (i.e., shape re-ordered so that fastest_axis is last).
    transposed_shape = [shape[a] for a in all_axes]
    temp = vec.reshape(transposed_shape)

    # Now invert that transpose by applying the reverse permutation.
    # If we originally did np.transpose(arr, axes=all_axes),
    # we now do np.transpose(..., inv_perm) with inv_perm being
    # the inverse of `all_axes`.
    all_axes = list_to_array(all_axes, nx=nx)
    inv_perm = nx.argsort(all_axes)  # which axes go back to their original position
    arr = nx.transpose(temp, axes=tuple(inv_perm))

    return arr


def vectorize_VF(V, F):
    """
    Vectorise the staggered-grid variables stored in ``V`` together with the
    1-D array ``F`` into a single flat vector.

    Parameters
    ----------
    V : Svar
        Staggered variable containing
        * a list ``V.D`` of ``V.N`` flux arrays, and
        * a cell-centred density array ``V.Z``.
        Each array has shape corresponding to the computational grid;
        the flux arrays may be staggered along their own axes.
    F : np.ndarray
        1-D array holding an additional field (e.g. external force or source).
        Its length does **not** have to equal any particular power-of-grid size.

    Returns
    -------
    np.ndarray
        1-D vector obtained by concatenating

        1. ``flatten_array(V.D[i], fastest_axis=i)`` for ``i = 0 ... V.N-1``
        2. ``flatten_array(V.Z, fastest_axis=V.N-1)``
        3. ``F``

        The length is ``(V.N + 1) * np.prod(V.Z.shape) + F.size``.
    """
    # Get the backend module
    nx = get_backend_ext(F)
    res = []

    for i in range(V.N):
        res.append(flatten_array(V.D[i], fastest_axis=i))
    res.append(flatten_array(V.Z, fastest_axis=V.N - 1))
    res.append(F)

    # Concatenate the two vectors
    vec = nx.concatenate(res)
    return vec


def vectorize_VF_multiF(V, F_list):
    """
    Flatten V and *all* F_i into one long 1-D array.

    Parameters
    ----------
    V : Svar
    F_list : Sequence[np.ndarray]   #  each of length N0

    Returns
    -------
        1-D array of length ``(V-part) + n * N0``
    """
    nx = get_backend_ext(F_list[0])
    if not isinstance(F_list, list):
        raise TypeError("F_list must be a list of arrays")
    pieces = []

    # 1. fluxes
    for i in range(V.N):
        pieces.append(flatten_array(V.D[i], fastest_axis=i))

    # 2. centred density
    pieces.append(flatten_array(V.Z, fastest_axis=V.N - 1))

    # 3. all right-hand sides
    pieces.extend(F_list)

    return nx.concatenate(pieces)


def unvectorize_VF(vec, F_shape, cs, ll):
    """
    Unvectorize the V and F variables from a single 1D vector.

    Parameters
    ----------
    vec : np.ndarray
        A 1D vector of shape (2 * N0 * N1 * ... * N_{N - 1}).
    F_shape : int
        The shape of the F variable (N_{N - 1}).
    cs : tuple of ints
        The size of the centered grid.
    ll : tuple of floats
        The length scales of the domain.

    Returns
    -------
    V : Svar
        The staggered variable V.
    F : np.ndarray
        The 1D array F.
    """
    # Get the backend module
    nx = get_backend_ext(vec)

    # Split the vector into V and F
    D = [nx.zeros(cs, type_as=vec) for _ in range(len(cs))]
    Z = nx.zeros(cs, type_as=vec)
    V = grids.Cvar(cs, ll, D, Z)
    F = vec[-F_shape:] if F_shape > 0 else None  # noqa: E203
    vec = (
        vec[:-F_shape].reshape(-1, math.prod(cs))
        if F_shape > 0
        else vec.reshape(-1, math.prod(cs))
    )

    for i in range(V.N):
        V.D[i] = unflatten_array(vec[i], shape=cs, fastest_axis=i)
    V.Z = unflatten_array(vec[V.N], shape=cs, fastest_axis=V.N - 1)

    return V, F


def unvectorize_VF_multiF(vec, F_shapes, cs, ll):
    """
    Inverse of vectorize_VF_multiF.

    F_shapes : Sequence[int]   # e.g. [N0]*n
    """
    nx = get_backend_ext(vec)
    F_list = []

    # 1. peel off the F blocks (last to first for convenience)
    tail = vec
    for length in reversed(F_shapes):
        F_list.append(tail[-length[0] :])  # noqa: E203
        tail = tail[: -length[0]]
    F_list.reverse()  # restore original order

    # 2. the remainder is V-part  (exactly as before)
    V_part = tail
    V, _dump = unvectorize_VF(V_part, 0, cs, ll)  # reuse your old helper

    return V, F_list


def build_H_operator_matrix(H, dx: float, fastest_axis: int = 0):
    """
    Construct the matrix M (size N0 x (N0*N1*...*N_{n-1})) that performs the operation:

        (M * flatten(rho))[i0] = sum_{i1,...,i_{n-1}} [ rho[i0,i1,...] * H[i0,i1,...] ] * dx

    This is consistent with the same flattening order used in flatten_array(..., fastest_axis).

    Parameters
    ----------
    H : np.ndarray
        N-dimensional array of shape (N0, N1, ..., N_{n-1}).
    dx : float
        Scalar factor that multiplies each H-element in the operator.
    fastest_axis : int, default=0
        The axis that is used as the fastest in flatten_array/unflatten_array.

    Returns
    -------
    M : np.ndarray
        A 2D matrix of shape (N0, N0*N1*...*N_{n-1}), representing the linear operator.
    """
    shape = H.shape
    dims = len(shape)
    n0 = shape[0]  # The "first" dimension
    total_size = math.prod(shape)  # Number of elements total
    nx = get_backend_ext(H)

    # We'll build the matrix directly.
    # M[row=i0, col] = H[i0, i1,...] * dx if that col corresponds to (i0,i1,...) in flattening order
    M = nx.zeros((n0, total_size), type_as=H)

    # We need to decode the multi-index from each column index
    # "Inverse" of flatten_array logic.
    nd = len(shape)
    # We'll make a helper to do the multi-index decoding:

    def index_to_multiidx(flat_idx: int) -> tuple:
        """
        Convert a single integer index `flat_idx` back into
        an (i0, i1, ..., i_{n-1}) multi-index consistent
        with flatten_array(..., fastest_axis).
        """
        # Step 1: figure out the permutation of axes
        all_axes = list(range(nd))
        all_axes[fastest_axis], all_axes[-1] = all_axes[-1], all_axes[fastest_axis]

        # Step 2: decode as if shape is reordered by `all_axes`
        reordered_shape = [shape[a] for a in all_axes]

        # We'll decode in standard row-major for that reordered shape.
        # i.e. if we label them as [r0, r1, ..., r_{n-1}] = reordered_shape,
        # then r_{n-1} is the fastest varying in normal flattening.
        # We'll accumulate each coordinate in that order, then invert.
        coords_reordered = []
        tmp = flat_idx
        for dim_size in reversed(reordered_shape):
            coord = tmp % dim_size
            tmp //= dim_size
            coords_reordered.append(coord)
        coords_reordered.reverse()  # because we took them from last to first

        # coords_reordered now corresponds to axes in `all_axes` order.
        # We must invert that permutation to recover the (i0, i1, ..., i_{n-1}) standard indexing.
        # That is, if all_axes = [A0, A1, ..., A_{n-1}],
        # then coords_reordered[j] = i_{A_j}.
        # We want i_j = coords_reordered[index_of_j_in_all_axes].
        full_coords = [None] * nd
        for axis_idx, axis_number in enumerate(all_axes):
            full_coords[axis_number] = coords_reordered[axis_idx]

        return tuple(full_coords)

    # Now fill the matrix:
    for col in range(total_size):
        # decode multi-index
        multi_idx = index_to_multiidx(col)
        i0 = multi_idx[0]

        # place H[i0,i1,...] * dx at M[i0, col]
        M[i0, col] = H[multi_idx] * dx

    # Concatenate zero blocks
    M = nx.concatenate([M] + ([nx.zeros((n0, total_size))] * dims), axis=1)

    return nx.tocsr(M)  # This returns dense matrix for torch backend


def build_H_operator_matrix_multi(H_list, dx, fastest_axis=0):
    """Return the sparse matrix of size ``(n * N0) x (3 * N0 * N1)``.

    The block stacks all ``n`` constraints in one shot.
    """
    nx = get_backend_ext(H_list[0])
    if nx == TorchBackend_ext:
        # Torch backend does not support sparse matrices
        raise NotImplementedError(
            "Affine constraints with Torch backend are not supported yet."
        )

    blocks = [
        build_H_operator_matrix(H_i, dx, fastest_axis) for H_i in H_list
    ]  # each is (N0) x (3 * N0 * N1)
    conc = sps.vstack(blocks)
    return conc  # shape = n * N0 rows


def I_of_N(N):
    """
    Construct the matrix I(N) of shape (N, N+1) where each row has two 0.5 entries:
      Row k has [0, ..., 0, 0.5, 0.5, 0, ..., 0]
                        ^      ^
                        k      k+1
    """
    M = np.zeros((N, N + 1))
    for k in range(N):
        M[k, k] = 0.5
        M[k, k + 1] = 0.5
    return M


def build_big_block(N_list: list, nx: Backend, periodic=False):
    """
    1) Given N_list = [N0, ..., N_N], build [I(N_0), ..., I(N_N)].
    2) For each j, build Q[j] = I(N_j) @ I(N_j).T + Identity(N_j).
    3) Form a block-diagonal matrix of the Q[j] blocks, where Q[j] is repeated
       (prod_over_i(N_i) / N_j) times.
    4) Add 2 * Identity(N_N) to the last block.
    """
    # Step 1 & 2: build each Q_j
    Q_list = []
    for i in range(len(N_list)):
        N_j = N_list[i]
        I_Nj = (
            I_of_N(N_j)
            if not periodic or i == 0
            else periodic_interpolation_matrix(N_j, nx)
        )
        Q_j = I_Nj @ I_Nj.T + np.eye(N_j)  # N_j x N_j
        Q_list.append(sps.csr_matrix(Q_j))

    # Compute the product of all N_j
    prod = math.prod(N_list)

    # Add 2 * the identity matrix for the source term
    Q_list.append(sps.csr_matrix(2 * np.eye(N_list[-1])))
    N_list.append(N_list[-1])

    # Step 3: repeat each Q_j the correct number of times and build block diagonal
    blocks = []
    for N_j, Q_j in zip(N_list, Q_list):
        repeats = prod // N_j  # integer division
        blocks.extend([Q_j] * repeats)

    # Build the final block-diagonal matrix
    big_block = sps.block_diag(blocks, format="csr")
    N_list.pop()  # Remove the last element (the source term)
    return nx.tocsr(nx.from_numpy(big_block))


def precomputePPT(H, cs, ll, periodic=False):
    """Precompute the matrix P^T P where P is the interpolation operator."""
    if not isinstance(H, list):
        H = [H]
    nx = get_backend_ext(*H)
    H_numpy = [nx.to_numpy(H_i) for H_i in H]
    numpy_back = NumpyBackend_ext()
    II_block = build_big_block(cs, numpy_back, periodic=periodic)
    H_op = build_H_operator_matrix_multi(
        H_numpy, math.prod(ll[1:]) / math.prod(cs[1:]), fastest_axis=0
    )
    # Check shapes before constructing the block
    rows_II, cols_II = II_block.shape
    rows_H, cols_H = H_op.shape
    rows_HT, cols_HT = H_op.T.shape
    H_opH_opT = H_op @ H_op.T
    rows_HopHopT, cols_HopHopT = H_opH_opT.shape
    # For np.block([[A, B],[C, D]]), we need:
    # 1) A.shape[0] == B.shape[0]  (top row blocks must align in rows)
    # 2) C.shape[0] == D.shape[0]  (bottom row blocks must align in rows)
    # 3) A.shape[1] == C.shape[1]  (left column blocks must align in columns)
    # 4) B.shape[1] == D.shape[1]  (right column blocks must align in columns)
    assert rows_II == rows_HT, "Top row blocks do not align in rows: {} vs {}".format(
        rows_II, rows_HT
    )
    assert (
        rows_H == rows_HopHopT
    ), "Bottom row blocks do not align in rows: {} vs {}".format(rows_H, rows_HopHopT)
    assert (
        cols_II == cols_H
    ), "Left column blocks do not align in columns: {} vs {}".format(cols_II, cols_H)
    assert (
        cols_HT == cols_HopHopT
    ), "Right column blocks do not align in columns: {} vs {}".format(
        cols_HT, cols_HopHopT
    )

    PPT = sps.block_array([[II_block, -H_op.T], [-H_op, H_opH_opT]], format="csr")
    return nx.tocsr(nx.from_numpy(PPT))


def projinterp_constraint_big_matrix(
    dest: grids.CSvar, x: grids.CSvar, solve: callable, H, F, log=None, periodic=False
):
    r"""Project using the explicit ``PP^T`` block system built by :func:`precomputePPT`.

    Given ``x = (U, V)``, we assemble the block matrix

    .. math::
        P = \begin{pmatrix} I & -\mathrm{Id} \\ 0 & H \end{pmatrix},

    where ``I`` is the interpolation operator and ``H`` encodes the affine
    constraints.  The projection solves the linear system

    .. math::
        (P P^T) y = (\mathcal{O}, F) - P(U, V),

    followed by the update ``(U', V') = (U, V) + P^T y``.  The sparse system
    ``PP^T`` is factorised once via :func:`precomputePPT` and a ``solve`` callback
    applies ``(PP^T)^{-1}`` to vectors during the iterations.

    Args:
        dest (grids.CSvar): Output pair ``(U', V')`` after projection.
        x (grids.CSvar): Input pair ``(U, V)`` before projection.
        solve (Callable[[array-like], array-like]): Linear solver that applies
            ``(PP^T)^{-1}`` to a vectorised residual.
        H (Sequence[array-like] | array-like): Constraint operators for each
            inequality block.
        F (Sequence[array-like] | array-like): Right-hand sides matching ``H``.
        log (dict, optional): Diagnostics dictionary mirroring the keys used in
            :func:`projinterp_constraint_`.
        periodic (bool, optional): Whether spatial axes use periodic staggering
            (affects the interpolation operator ``I``).
    """
    if log is None:
        log = {
            "pre_lambda": [],
            "lambda": [],
            "first_order_condition": [],
            "U_prime_interp": [],
            "HQHlambda": [],
            "lambda_eqn": [],
        }
    nx = x.nx
    if not isinstance(H, list):
        H = [H]
    if not isinstance(F, list):
        F = [F]
    if len(H) != len(F):
        raise ValueError("H and F must have the same length.")
    if not isinstance(nx, NumpyBackend_ext):
        raise ValueError("The backend must be numpy. {} is not supported.".format(nx))
    # Calculate (O, F) - P(U, V) = (V - I(U), F - HV)
    V_minus_IU = x.V - grids.interp(x.U, periodic=periodic)
    F_minus_HV = [
        F_i
        - nx.sum(x.V.D[0] * H_i, axis=tuple(range(1, H_i.ndim)))
        * math.prod(x.ll[1:])
        / math.prod(x.cs[1:])
        for F_i, H_i in zip(F, H)
    ]

    # Solve the linear system to calculate (P P^T)^{-1} ((O, F) - P(U, V))
    vec = vectorize_VF_multiF(V_minus_IU, F_minus_HV)
    vec = solve(vec)
    new_V, new_F = unvectorize_VF_multiF(
        vec, tuple([F_i.shape for F_i in F]), x.cs, x.ll
    )

    # Calculate (U', V') =  P^T(new_V, new_F) = (I* new_V, - new_V + H^*new_F )
    dest.U = grids.interpT(new_V, periodic=periodic)
    Hstar_new_F = sum(
        H_i
        * new_F_i[(slice(None),) + (None,) * (H_i.ndim - 1)]
        * (math.prod(x.ll[1:]) / math.prod(x.cs[1:]))
        for H_i, new_F_i in zip(H, new_F)
    )
    dest.V = (-1.0) * new_V
    dest.V.D[0] += Hstar_new_F

    # Calculate (U', V') = (U, V) + P^T (P P^T)^{-1} ((O, F) - P(U, V))
    dest.U += x.U
    dest.V += x.V


def project_constraint_inequality_single_(
    dest: grids.CSvar, v: grids.CSvar, Hs: List, GL, GU
):
    r"""Project ``v`` onto the constraint ``GL(t) <= H(t, v) <= GU(t)``.

    Writing ``v = (rho, omega, zeta)`` and

    .. math::
        H(t, v) = \int H_\rho(t, x) \rho(t, x)\,dx
            + \int \langle H_\omega(t, x), \omega(t, x) \rangle\,dx
            + \int H_\zeta(t, x) \zeta(t, x)\,dx,

    the projection updates

    .. math::
        v \leftarrow v - \sum_{i=0}^{T-1} \lambda_i(v) \frac{H(t_i, \cdot)}{\|H(t_i, \cdot)\|^2},

    with coefficients

    .. math::
        \lambda_i(v) =
            \begin{cases}
                0 & \text{if } GL(t_i) \le H(t_i, v) \le GU(t_i),\\
                H(t_i, v) - GL(t_i) & \text{if } H(t_i, v) < GL(t_i),\\
                H(t_i, v) - GU(t_i) & \text{if } H(t_i, v) > GU(t_i).
            \end{cases}

    Args:
        dest (grids.CSvar): Destination variable overwritten with the projection.
        v (grids.CSvar): Input variable to be projected.
        Hs (list[array-like]): Kernels ``[H_rho, H_omega0, H_omega1, ..., H_zeta]``.
        GL (array-like): Lower bounds of the constraint.
        GU (array-like): Upper bounds of the constraint.
    """
    if not isinstance(Hs, list):
        raise ValueError("Hs must be a list of arrays.")
    if len(Hs) != len(v.cs) + 1:
        raise ValueError(
            f"Hs must have exactly {len(v.cs) + 1} elements: H_rho, H_omega, H_zeta. We got: {len(Hs)} elements."
        )
    if any(H.shape != v.cs for H in Hs):
        raise ValueError("All H functions must have the same shape as cs.")
    if len(GL) != len(GU):
        raise ValueError("GL and GU must have the same length.")
    T = len(GL)  # time steps
    # print(
    #    "NaNs in input variable:", [np.isnan(D).any() for D in v.V.D + [v.V.Z]]
    # )  # Check for NaNs in the input variable

    dx = math.prod(v.ll[1:]) / math.prod(v.cs[1:])  # grid spacing
    nx = get_backend_ext(GL, GU, *Hs)
    ndim = len(v.cs)
    # Calculate the lambda_i(v) for each time step
    LHS = (
        sum(
            nx.sum(H_i * v_i, axis=tuple(range(1, ndim)))
            for H_i, v_i in zip(Hs, v.V.D + [v.V.Z])
        )
        * dx
    )
    assert LHS.shape == (
        T,
    ), f"LHS must be a 1D array of shape (T,). We got: {LHS.shape}"
    # LHS is now a 1D array of shape (T,)
    lamda = nx.zeros(T, type_as=GL)
    lamda = nx.where(LHS < GL, LHS - GL, lamda)  # if LHS < GL, then lambda = LHS - GL
    lamda = nx.where(LHS > GU, LHS - GU, lamda)  # if LHS > GU, then lambda = LHS - GU
    # if GL <= LHS <= GU, then lambda = 0 (already set)
    # Now we have the lambda_i(v) for each time step
    # Calculate the projection
    H_norm_sq = sum(nx.sum(H_i**2, axis=tuple(range(1, ndim))) for H_i in Hs) * dx
    assert H_norm_sq.shape == (T,), "H_norm_sq must be a 1D array of shape (T,)"
    multiplier = lamda / nx.maximum(H_norm_sq, 1e-12)
    # --- projection ---------------------------------------------------
    shape_exp = (-1,) + (1,) * (ndim - 1)
    exp_mult = multiplier.reshape(shape_exp)

    dest.U = v.U.copy()  # unchanged component
    dest.V.D = [v_i - exp_mult * H_i for v_i, H_i in zip(v.V.D, Hs[:-1])]
    dest.V.Z = v.V.Z - exp_mult * Hs[-1]


def stepDR(
    w: grids.CSvar, x: grids.CSvar, y: grids.CSvar, z: grids.CSvar, prox1, prox2, alpha
):
    """Apply one step of the Douglas-Rachford algorithm to the variables
    w, x, y, and z."""
    # Step 1: Update x based on z and w
    x = 2 * z - w

    # Step 2: Apply proximal operator 1 to x, updating y
    prox1(y, x)

    # Step 3: Update w with step size alpha
    w += alpha * (y - z)

    # Step 4: Apply proximal operator 2 to w, updating z
    prox2(z, w)

    return w, x, y, z


def stepPPXA(
    x: grids.CSvar,
    ys: List[grids.CSvar],
    pis: List[grids.CSvar],
    proxs: List[callable],
    alpha: float,
):
    """Perform one step of the PPXA algorithm for solving the problem
    min_{x} sum_{i} f_i(x).

    Args:
        x (array): The current variable to be updated.
        ys (list of arrays): The list of variables y_i to be updated.
        pis (list of arrays): The list of proximal outputs pi_i.
        proxs (list of callable): The proximal operators for each f_i.
        alpha (float): The step size for the update.
    """
    # Step 1: Calculate proximal operators
    for prox, pi, y in zip(proxs, pis, ys):
        prox(pi, y)

    # Step 2: Calculate the mean of the proximal outputs
    p = sum(pis[1:], start=pis[0]) * (1.0 / len(pis))

    # Step 3: Update each variable y with the step size alpha
    for y, pi in zip(ys, pis):
        y += alpha * (2 * p - x - pi)

    # Step 4: Update x with the mean of the proximal outputs
    x += alpha * (p - x)

    return x, ys, pis


def computeGeodesic_equality(
    rho0,
    rho1,
    T,
    ll,
    H=None,
    F=None,
    p=2.0,
    q=2.0,
    delta=1.0,
    niter=1000,
    big_matrix=True,
    periodic=False,
    alpha=None,
    gamma=None,
    verbose=False,
    log=None,
    init=None,
    U=None,
    V=None,
):
    r"""Compute a dynamic unbalanced OT geodesic via Douglas-Rachford splitting.
    The solver is only compatible with an equality constraint of the form
    \int_{\Omega} H(t, x) \rho(t, x) dx = F(t), but it is often faster than the main solver.

    Args:
        rho0 (array-like): Initial density.
        rho1 (array-like): Target density.
        T (array-like): Number of time points on the centred grid.
        ll (tuple[float, ...]): Physical lengths of each spatial axis.
        H (array-like, optional): Constraint operator.  When omitted the solver
            solves the unconstrained dynamic OT problem.
        F (array-like, optional): Affine right-hand side associated with ``H``.
        p (float, optional): Transport exponent (``1`` or ``2``).  Defaults to ``2``.
        q (float, optional): Source exponent controlling mass change. Defaults to ``2``.
        delta (float, optional): Interpolation parameter controlling how much mass
            creation/destruction is penalized (larger values impose stronger
            penalties).
        niter (int, optional): Maximum Douglas-Rachford iterations.
        big_matrix (bool, optional): If ``True`` use the explicit ``PP^T`` block
            factorisation; otherwise rely on precomputed ``HQH``.
        periodic (bool, optional): Enable periodic boundary conditions in space.
        alpha (float, optional): Relaxation parameter for the DR update.  When
            ``None`` a heuristic default is chosen from the input data.
        gamma (float, optional): Step size of the proximal map associated with ``F``.
        verbose (bool, optional): Emit textual progress every 1% of the iterations.
        log (dict, optional): Diagnostics dictionary populated by the proximal
            operators.
        init (str, optional): Initialisation strategy (``None`` for linear
            interpolation, ``"fisher-rao"``, or ``"manual"`` with ``U``/``V``).
        U (array-like, optional): Initial ``U`` used when ``init == "manual"``.
        V (array-like, optional): Initial ``V`` used when ``init == "manual"``.

    Returns:
        tuple[grids.CSvar, tuple]: The converged variable ``z`` together with
        diagnostic arrays ``(Flist, Clist, Ilist, HFlist)`` capturing the energy,
        continuity residual, interpolation residual, and (if ``H`` is provided)
        constraint violation per iteration.
    """
    assert delta > 0, "Delta must be positive"
    source = q >= 1.0  # Check if source problem

    nx = get_backend_ext(rho0, rho1)

    def prox1(y: grids.CSvar, x: grids.CSvar, source, gamma, p, q, periodic=False):
        projCE_(
            y.U, x.U, rho0 * delta**rho0.ndim, rho1 * delta**rho0.ndim, source, periodic
        )
        proxF_(y.V, x.V, gamma, p, q)

    def prox2(
        y: grids.CSvar,
        x: grids.CSvar,
        Q,
        solve=None,
        HQH=None,
        H=None,
        F=None,
        big_matrix=False,
    ):
        if H is None or F is None:
            projinterp_(y, x, Q, log)
        else:
            if big_matrix:
                projinterp_constraint_big_matrix(y, x, solve, H, F, log, periodic)
            else:
                projinterp_constraint_(y, x, Q, HQH, H, F, log)

    # Adjust mass to match if not a source problem
    if q < 1.0:
        if verbose:
            print("Computing geodesic for standard optimal transport...")
        rho1 *= nx.sum(rho0) / nx.sum(rho1)
        delta = 1.0  # Ensure delta is set correctly for non-source problems
        if alpha is None:
            alpha = 1.8
        if gamma is None:
            gamma = max(nx.max(rho0), nx.max(rho1)) / 2
    else:
        if H is None or F is None:
            if verbose:
                print("Computing a geodesic for optimal transport with source...")
        else:
            if verbose:
                print(
                    "Computing a geodesic for optimal transport with source and constraint..."
                )
        if alpha is None:
            alpha = 1.8
        if gamma is None:
            gamma = delta**rho0.ndim * max(nx.max(rho0), nx.max(rho1)) / 15

    # Initialization
    w, x, y, z = [
        grids.CSvar(rho0, rho1, T, ll, init, U, V, periodic) for _ in range(4)
    ]

    # Change of variable for scale adjustment
    for var in [w, x, y, z]:
        var.dilate_grid(1 / delta)
        var.rho1 *= delta**rho0.ndim
        var.rho0 *= delta**rho0.ndim

    # Precompute projection interpolation operators if needed
    Q = precomputeProjInterp(x.cs, rho0, rho1, periodic)
    HQH = (
        precomputeHQH(Q[0], H, x.cs, x.ll)
        if not big_matrix and (H is not None)
        else None
    )
    PPT = (
        precomputePPT(H, list(x.cs), x.ll, periodic=periodic)
        if big_matrix and (H is not None)
        else None
    )
    solve = spsla.factorized(PPT) if PPT is not None else None

    Flist, Clist, Ilist = (
        nx.zeros(niter, type_as=rho0),
        nx.zeros(niter, type_as=rho0),
        nx.zeros(niter, type_as=rho0),
    )
    HFlist = nx.zeros(niter, type_as=rho0) if H is not None else None

    for i in range(niter):
        # print("Iteration:", i)
        if i % (niter // 100) == 0:
            if verbose:
                print(f"\rProgress: {i // (niter // 100)}%", end="")

        w, x, y, z = stepDR(
            w,
            x,
            y,
            z,
            lambda y, x: prox1(y, x, source, gamma, p, q, periodic),
            lambda y, x: prox2(
                y,
                x,
                Q,
                solve,
                HQH,
                H,
                F,
                big_matrix,
            ),
            alpha,
        )

        Flist[i] = z.energy(delta, p, q)
        Clist[i] = z.dist_from_CE()
        Ilist[i] = z.dist_from_interp()
        if H is not None:
            HFlist[i] = sum(z.dist_from_constraint(H, F))

    # Final projection and positive density adjustment
    projCE_(
        z.U, z.U, rho0 * delta**rho0.ndim, rho1 * delta**rho0.ndim, source, periodic
    )
    z.proj_positive()
    z.dilate_grid(delta)  # Adjust back to original scale
    z.interp_()  # Final interpolation adjustment

    if verbose:
        print("\nDone.")

    return z, (Flist, Clist, Ilist, HFlist)


def computeGeodesic(
    rho0,
    rho1,
    T,
    ll,
    H=None,
    GL=None,
    GU=None,
    p=2.0,
    q=2.0,
    delta=1.0,
    niter=1000,
    periodic=False,
    alpha=None,
    gamma=None,
    verbose=False,
    log=None,
    init=None,
    U=None,
    V=None,
    track_terminal_distance=False,
):
    """Main PPXA-based solver for dynamic unbalanced optimal transport.

    Args:
        rho0 (array-like): Initial density.
        rho1 (array-like): Target density.
        T (array-like): Number of time points on the centred grid.
        ll (tuple[float, ...]): Physical lengths per axis.
        H (Sequence[Sequence[array-like]] | None): Collection of inequality
            kernels grouped per constraint block.  Each block mirrors the layout
            ``[H_rho, H_omega..., H_zeta]``.
        GL (Sequence[array-like] | None): Lower bounds for each constraint block.
        GU (Sequence[array-like] | None): Upper bounds for each constraint block.
        p (float, optional): Transport exponent. Defaults to ``2``.
        q (float, optional): Source exponent. Defaults to ``2``.
        delta (float, optional): Interpolation parameter controlling how much mass
            creation/destruction is penalized (larger values impose stronger
            penalties).
        niter (int, optional): Maximum PPXA iterations.
        periodic (bool, optional): Enable periodic spatial boundary conditions.
        alpha (float, optional): Relaxation parameter (auto-selected when ``None``).
        gamma (float, optional): Step size for the kinetic-energy proximal map.
        verbose (bool, optional): Emit textual progress updates.
        log (dict, optional): Diagnostics dictionary shared by proximal operators.
        init (str, optional): Initialisation strategy.
        U (array-like, optional): Manual initial ``U`` when ``init == "manual"``.
        V (array-like, optional): Manual initial ``V`` when ``init == "manual"``.
        track_terminal_distance (bool, optional): If ``True``, compute and return
            the distance of each iterate x to the last iterate.

    Returns:
        tuple[grids.CSvar, tuple]: Final iterate ``x`` and diagnostic arrays
        ``(Flist, Clist, Ilist, HFlist)``.
    """
    assert delta > 0, "Delta must be positive"
    source = q >= 1.0  # Check if source problem

    nx = get_backend_ext(rho0, rho1)

    def prox1(y: grids.CSvar, x: grids.CSvar, source, gamma, p, q, periodic=False):
        projCE_(
            y.U, x.U, rho0 * delta**rho0.ndim, rho1 * delta**rho0.ndim, source, periodic
        )
        proxF_(y.V, x.V, gamma, p, q)

    def prox2(
        y: grids.CSvar,
        x: grids.CSvar,
        Q,
    ):
        projinterp_(y, x, Q, log)

    # Define the list of proximal operators
    proxs = [
        (lambda pi, y: prox1(pi, y, source, gamma, p, q, periodic)),
        (lambda pi, y: prox2(pi, y, Q)),
    ]
    if H is not None or GL is not None or GU is not None:
        if not (isinstance(H, list) and isinstance(GL, list) and isinstance(GU, list)):
            raise ValueError("H, GL, and GU must be lists.")
        if not (len(H) == len(GL) == len(GU)):
            raise ValueError("H, GL, and GU must have the same length.")
        for Hj, GLj, GUj in zip(H, GL, GU):
            proxs.append(
                (
                    lambda pi, y, Hj=Hj, GLj=GLj, GUj=GUj: project_constraint_inequality_single_(
                        pi, y, Hj, GLj, GUj
                    )
                )
            )

    # Adjust mass to match if not a source problem
    if q < 1.0:
        if verbose:
            print("Computing geodesic for standard optimal transport...")
        rho1 *= nx.sum(rho0) / nx.sum(rho1)
        delta = 1.0  # Ensure delta is set correctly for non-source problems
        if alpha is None:
            alpha = 1.8
        if gamma is None:
            gamma = max(nx.max(rho0), nx.max(rho1)) / 2
    else:
        if H is None or GL is None or GU is None:
            if verbose:
                print("Computing a geodesic for optimal transport with source...")
        else:
            if verbose:
                print(
                    "Computing a geodesic for optimal transport with source and constraint... (including inequality constraints)"
                )
        if alpha is None:
            alpha = 1.8
        if gamma is None:
            gamma = delta**rho0.ndim * max(nx.max(rho0), nx.max(rho1)) / 15

    # Initialization
    n_constraints = len(H) if H is not None else 0
    K = 2 + n_constraints  # Number of proximal operators
    ys = [grids.CSvar(rho0, rho1, T, ll, init, U, V, periodic) for _ in range(K)]
    pis = [grids.CSvar(rho0, rho1, T, ll, init, U, V, periodic) for _ in range(K)]
    x = sum(ys[1:], start=ys[0]) * (1.0 / len(ys))  # Average of the initial variables

    # Change of variable for scale adjustment
    for var in [x] + ys + pis:
        var.dilate_grid(1 / delta)
        var.rho1 *= delta**rho0.ndim
        var.rho0 *= delta**rho0.ndim

    # Precompute projection interpolation operators if needed
    Q = precomputeProjInterp(x.cs, rho0, rho1, periodic)
    Flist, Clist, Ilist = (
        nx.zeros(niter, type_as=rho0),
        nx.zeros(niter, type_as=rho0),
        nx.zeros(niter, type_as=rho0),
    )

    Dlist = None
    if track_terminal_distance:
        Dlist = nx.zeros(niter, type_as=rho0)
        xlist = []

    for i in range(niter):
        if i % (niter // 100) == 0:
            if verbose:
                print(f"\rProgress: {i // (niter // 100)}%", end="")
        x, ys, pis = stepPPXA(
            x,
            ys,
            pis,
            proxs,
            alpha,
        )

        Flist[i] = x.energy(delta, p, q)
        Clist[i] = x.dist_from_CE()
        Ilist[i] = x.dist_from_interp()

        if track_terminal_distance:
            xlist.append(x.copy())

    # Final projection and positive density adjustment
    projCE_(
        x.U, x.U, rho0 * delta**rho0.ndim, rho1 * delta**rho0.ndim, source, periodic
    )
    x.proj_positive()
    x.dilate_grid(delta)  # Adjust back to original scale
    x.interp_()  # Final interpolation adjustment

    if verbose:
        print("\nDone.")

    if track_terminal_distance:
        Dlist = nx.zeros(niter, type_as=rho0)
        x_final = xlist[-1]
        for i in range(niter):
            Dlist[i] = (xlist[i] - x_final).norm()

    return x, (Flist, Clist, Ilist, Dlist)
