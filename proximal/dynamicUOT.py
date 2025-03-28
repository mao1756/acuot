import proximal.grids as grids
from proximal.backend_extension import get_backend_ext
from proximal.backend_extension import NumpyBackend_ext
from ot.backend import Backend
from ot.utils import list_to_array
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import math


def root(a, b, c, d, nx: Backend):
    """Compute the largest real root of a cubic polynomial ax^3 + bx^2 + cx + d.
    If a, b, c, and d are arrays, the function computes the root elementwise.

    Args:
        a (array): The coefficient of x^3.
        b (array): The coefficient of x^2.
        c (array): The coefficient of x.
        d (array): The constant term.
        nx (module): The backend module used for computation such as numpy or torch.

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
    """Given a list of arrays of the same shape, return the elementwise L2 norm."""
    return nx.sqrt(sum(m**2 for m in M))


def proxA_(dest: grids.Cvar, M, gamma):
    """In-place calculation of proximal operator for sum abs (M_i)

    I think this is actually calculating the proximal operator for the L2 norm
    i.e. sum abs(M_i)^2

    As this operator does not affect WFR, I keep it as it is for now.

    """
    softth = dest.nx.maximum(1 - gamma / mdl(M, dest.nx), 0.0)
    for k in range(len(M)):
        dest[k] = softth * M[k]


def proxB_(destR, destM: list, R, M: list, gamma: float, nx: Backend):
    """In-place calculation of proximal operator for sum |M_i|^2/R_i."""
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
    "Return prox_F(V) where F is the energy functional and V on centered grid"
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
    """Solve Δu+f=0(source=False) or Δu-u+f=0 on the centered grid.
    The BC is Neumann BC defined on the staggered grid.

    f will be overwritten with the solution in-place.

    Args:
        f (array): The function f on the centered grid.
        ll (tuple): The length scales of the domain.
        source (bool): True if source problem.
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

    # dctn
    """
    renorm =  math.prod(N) * (2**d)
    f[...] = nx.dctn(f, axes=range(d))
    f /= -D * renorm
    f[...] = nx.idctn(f, axes=range(d))
    """
    # axis wise DCT
    for axe in range(d):
        f[...] = nx.dct(f, axis=axe, norm="ortho")
    f /= -D
    for axe in range(d):
        f[...] = nx.idct(f, axis=axe, norm="ortho")


def minus_interior_(dest, M, dpk, cs, dim):
    """ Subtract dpk from the interior of a staggered variable M and replaces the \
        interior of dest in place by the result.

    Args:
        dest (array): The destination array. It should have the same shape as M.
        M (array): The source array. It should be defined on a staggered grid. \
        That is, the shape should be (N0, N1, ..., N_k +1,  Nd) where k is the staggered\
        dimension.
        dpk (array): The array to subtract. It should have the shape\
        (N0, N1,..., N_k-1,..., Nd).
        cs (tuple): The size of the centered grid i.e. (N0, N1, ..., Nd).
        dim (int): The dimension along which to subtract. k in the above description.
    """
    assert dest.shape == M.shape, "Destination and source shapes must match"
    slices = [slice(None)] * M.ndim
    slices[dim] = slice(1, cs[dim])

    interior_diff = M[tuple(slices)] - dpk
    dest[tuple(slices)] = interior_diff


def projCE_(dest: grids.Svar, U: grids.Svar, rho_0, rho_1, source: bool):
    """ Given a staggered variable U, project it so that it satisfies the \
        continuity equation. The result is stored in dest in place.

    Args:
        dest (Svar): The destination variable.
        U (Svar): The source variable.
        rho_0 (array): The source density.
        rho_1 (array): The destination density.
        source (bool): True if we are solving a OT with source problem.
    """

    assert dest.ll == U.ll, "Destination and source lengths must match"

    U.proj_BC(rho_0, rho_1)
    p = -U.remainder_CE()
    poisson_(p, U.ll, source, dest.nx)

    for k in range(len(U.cs)):
        dpk = dest.nx.diff(p, axis=k) * U.cs[k] / U.ll[k]
        minus_interior_(dest.D[k], U.D[k], dpk, U.cs, k)

    dest.proj_BC(rho_0, rho_1)
    if source:
        dest.Z[...] = U.Z - p


def invQ_mul_A_(dest, Q, src, dim: int, nx: Backend):
    """Apply the inverse of Q to the input src along the specified dimension.

    This function modifies the array dest in-place and calculates
    dest[i1, i2, ... , :, i_n] = Q^-1 * src[i1,i2,... :, i_n] for all i1,...,in and :\
    is at the specified dimension. That is, it applies the inverse of Q to each 1D slice.

    Args:
        dest (array): The destination array.
        Q (array): The matrix Q to apply the inverse.
        src (array): The source array.
        dim (int): The dimension along which to apply the inverse.
        nx (module): The backend module used for computation such as numpy or torch.

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
    """Calculate the projection of the interpolation operator for x.

    Given the input x=(U,V), calculate the projection of the interpolation operator by
    U' = Q^-1(U+I*V) and V' = I(U')
    where I is the interpolation operator.
    Here, Q = Id+I*I. The result is stored in dest=(U',V').

    Args:
        dest (CSvar): The destination variable.
        x (CSvar): The input variable.
        Q (list): The tensor of Q matrices [Q1, Q2, ..., QN] where
        Qk = Id + I^T I such that I is the interpolation matrix
        (1/2, 1/2, .... 0)
        (0, 1/2, 1/2, ...,0)
        ...
        (0, ..., 1/2, 1/2)
        of size (x.cs[k]-1) x x.cs[k] for each dimension k.
        We will precompute these matrices for efficiency.
        noV (bool): If True, only calculate U' and ignore V'.
    """
    assert dest.ll == x.ll, "Length scales must match"

    x_U_copy = x.U.copy()  # Copy x.U since interpT_ will overwrite x.U if dest=x

    # Calculate I*V and store it in U'
    grids.interpT_(dest.U, x.V)
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
    """Calculate the projection of the interpolation and constraint operator for x.

    Given the input x=(U,V), calculate the projection of interpolation&constraint
    operator.
    We first calculate
    lambda = (H(Id+I^* I)^{-1}H*)^{-1}(H(Id+I^* I)^{-1}(U+I*V)-F)
    where H is the H function for the constraint and F is the right-hand side.
    Then, we calculate
    U' = (Id+I^* I)^{-1}(U+I*V-H^* lambda)
    V' = I(U) where I is the interpolation operator.
    The result is stored in dest=(U',V').

    Args:
        dest (CSvar): The destination variable.
        x (CSvar): The input variable.
        Q (list): The tensor of Q matrices [Q1, Q2, ..., QN] where\
        Qk = Id + I^T I such that I is the interpolation matrix.
        HQH (array of shape (cs[0], cs[0])): The matrix HQ^{-1}H* where H=hI and \
            h is the H function for the constraint.
        H (array of shape cs): The H function for the constraint.
        F (array of shape (cs[0],)): The right-hand side of the constraint.
        log (dict): The dictionary to store norms of pre_lambda, lambda, and the first \
            order condition. Assumed to have the keys 'pre_lambda', 'lambda', and \
            'first_order_condition' and the values are lists to store the norms.
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
    Hstar_lambda_copy = Hstar_lambda.copy()

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
    U_hstar_lambda = grids.CSvar(x.rho0, x.rho1, x.cs[0], x.ll)
    U_hstar_lambda.U.D[0] = Hstar_lambda_copy
    U_hstar_lambda = U_hstar_lambda.U

    # Calculate U'-U_copy + I*(V'-V_copy) + H^* lambda
    first_order_condition = dest.U - x.U + interpT_V_diff + U_hstar_lambda

    dens_norm = 0
    # Calculate the norm for each dimension
    for dens in first_order_condition.D:
        dens_norm += dest.nx.norm(dens) ** 2
    log["first_order_condition"].append(dest.nx.sqrt(dens_norm))


def projinterp_constraint_dykstra(
    dest: grids.CSvar, x: grids.CSvar, Q, HQH, H, F, eps=10e-3, maxiter=1000, log=None
):
    """Calculate the projection of the interpolation, constraint operator AND positivity for x.

    Given the input x=(U,V), calculate the projection of interpolation&constraint
    operator AND positivity constraint using the Dykstra algorithm. We alternatively apply the projection of the interpolation/constraint operator and the positivity constraint.

    Args:
        dest (CSvar): The destination variable.
        x (CSvar): The input variable.
        Q (list): The tensor of Q matrices [Q1, Q2, ..., QN] where\
        Qk = Id + I^T I such that I is the interpolation matrix.
        HQH (array of shape (cs[0], cs[0])): The matrix HQ^{-1}H* where H=hI and \
            h is the H function for the constraint.
        H (array of shape cs): The H function for the constraint.
        F (array of shape (cs[0],)): The right-hand side of the constraint.
        eps (float): The tolerance for the algorithm.
        maxiter (int): The maximum number of iterations for the algorithm.
        log (dict): The dictionary to store norms of pre_lambda, lambda, and the first \
            order condition. Assumed to have the keys 'pre_lambda', 'lambda', and \
            'first_order_condition' and the values are lists to store the norms.
    """
    p = grids.CSvar(x.rho0, x.rho1, x.cs[0], x.ll, init="zero")
    q = grids.CSvar(x.rho0, x.rho1, x.cs[0], x.ll, init="zero")
    y = grids.CSvar(x.rho0, x.rho1, x.cs[0], x.ll, init="zero")
    x_new = grids.CSvar(x.rho0, x.rho1, x.cs[0], x.ll, init="zero")
    tolerance = float("inf")
    niter = 0
    while tolerance > eps and niter < maxiter:
        niter += 1
        projinterp_constraint_(y, x + p, Q, HQH, H, F, log)
        p = x + p - y
        x_new = y + q
        x_new.proj_positive()
        q = y + q - x_new
        tolerance = (x - x_new).norm()
        x = x_new
        print(niter, tolerance)

    if niter == maxiter:
        print("Dykstra algorithm did not converge. Increase the number of iterations.")

    dest.U = x.U
    dest.V = x.V


def precomputeProjInterp(cs, rho0, rho1):
    B = []
    nx = get_backend_ext(rho0, rho1)
    for n in cs:
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
    """Precompute the matrix HQ^{-1}H* where H is the H function for the constraint
    and Q=Id+I*I.

    Args:
        Q (array): Q = Id + I^T I such that I is the interpolation matrix
        (1/2, 1/2, .... 0)
        (0, 1/2, 1/2, ...,0)
        ...
        (0, ..., 1/2, 1/2)
        of size T x T+1 where T is the size for the time dimension in the centered grid.
        We will precompute these matrices for efficiency.
        H (array): The H function for the constraint.
        nx (module): The backend module used for computation such as numpy or torch.
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
    Vectorize the V and F variables into a single 1D vector.

    Parameters
    ----------
    V : Svar
        The staggered variable V. The shape of each density is (N0, N1, ..., N_{N - 1}).
    F : np.ndarray
        N-dimensional array of shape (N_{N - 1}).

    Returns
    -------
    vec : np.ndarray
        A 1D vector of shape (2 * N0 * N1 * ... * N_{N - 1}).
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
        The N-dimensional array F.
    """
    # Get the backend module
    nx = get_backend_ext(vec)

    # Split the vector into V and F
    D = [nx.zeros(cs, type_as=vec) for _ in range(len(cs))]
    Z = nx.zeros(cs, type_as=vec)
    V = grids.Cvar(cs, ll, D, Z)
    F = vec[-F_shape:]  # noqa: E203
    vec = vec[:-F_shape].reshape(-1, math.prod(cs))

    for i in range(V.N):
        V.D[i] = unflatten_array(vec[i], shape=cs, fastest_axis=i)
    V.Z = unflatten_array(vec[V.N], shape=cs, fastest_axis=V.N - 1)

    return V, F


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
    fastest_axis : int, default=1
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


def build_big_block(N_list: list, nx: Backend):
    """
    1) Given N_list = [N0, ..., N_N], build [I(N_0), ..., I(N_N)].
    2) For each j, build Q[j] = I(N_j) @ I(N_j).T + Identity(N_j).
    3) Form a block-diagonal matrix of the Q[j] blocks, where Q[j] is repeated
       (prod_over_i(N_i) / N_j) times.
    4) Add 2 * Identity(N_N) to the last block.
    """
    # Step 1 & 2: build each Q_j
    Q_list = []
    for N_j in N_list:
        I_Nj = I_of_N(N_j)
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


def precomputePPT(H, cs, ll):
    nx = get_backend_ext(H)
    H_numpy = nx.to_numpy(H)
    numpy_back = NumpyBackend_ext()
    II_block = build_big_block(cs, numpy_back)
    H_op = build_H_operator_matrix(
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
    assert rows_II == rows_HT, "Top row blocks do not align in rows."
    assert rows_H == rows_HopHopT, "Bottom row blocks do not align in rows."
    assert cols_II == cols_H, "Left column blocks do not align in columns."
    assert cols_HT == cols_HopHopT, "Right column blocks do not align in columns."

    PPT = sps.block_array([[II_block, -H_op.T], [-H_op, H_opH_opT]], format="csr")
    return nx.tocsr(nx.from_numpy(PPT))


def projinterp_constraint_big_matrix(
    dest: grids.CSvar, x: grids.CSvar, solve: callable, H, F, log=None
):
    """Calculate the projection of the interpolation and the constraint operator by constructing a big matrix.

    Given the input x=(U,V), calculate the projection of interpolation and constraint
    operator. We consider the matrix
    P = (I, -Id)
        (0,  H)
    where I is the interpolation operator, Id is the identity for the centered variable and H is the H function for the constraint. By noting that the interpolation & the constraint operator can be
    written as the affine equation
    P(U, V) = (O, F) (O is the zero variable and F is the right-hand side of the constraint),
    we calculate the projection dest = (U', V') by the projection formula:
    (U', V') = (U, V) + P^T (P P^T)^{-1} ((O, F) - P(U, V))
        
    Args:
        dest (CSvar): The destination variable.
        x (CSvar): The input variable.
        P (array): The matrix P.
        PPT (callable): The "solve" function for the matrix PPT.
        H (array): The H function for the constraint.
        F (array of shape (cs[0],)): The right-hand side of the constraint.
        log (dict): The dictionary to store norms of pre_lambda, lambda, and the first \
            order condition. Assumed to have the keys 'pre_lambda', 'lambda', and \
            'first_order_condition' and the values are lists to store the norms.
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
    if not isinstance(nx, NumpyBackend_ext):
        raise ValueError("The backend must be numpy. {} is not supported.".format(nx))
    # Calculate (O, F) - P(U, V) = (V - I(U), F - HV)
    V_minus_IU = x.V - grids.interp(x.U)
    F_minus_HV = F - nx.sum(x.V.D[0] * H, axis=tuple(range(1, H.ndim))) * math.prod(
        x.ll[1:]
    ) / math.prod(x.cs[1:])

    # Solve the linear system to calculate (P P^T)^{-1} ((O, F) - P(U, V))
    vec = vectorize_VF(V_minus_IU, F_minus_HV)
    vec = solve(vec)
    new_V, new_F = unvectorize_VF(vec, F.shape[0], x.cs, x.ll)

    # Calculate (U', V') =  P^T(new_V, new_F) = (I* new_V, - new_V + H^*new_F )
    dest.U = grids.interpT(new_V)
    Hstar_new_F = (
        H
        * new_F[(slice(None),) + (None,) * (H.ndim - 1)]
        * (math.prod(x.ll[1:]) / math.prod(x.cs[1:]))
    )
    dest.V = (-1.0) * new_V
    dest.V.D[0] += Hstar_new_F

    # Calculate (U', V') = (U, V) + P^T (P P^T)^{-1} ((O, F) - P(U, V))
    dest.U += x.U
    dest.V += x.V


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


def computeGeodesic(
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
    dykstra=False,
    big_matrix=False,
    eps=10e-3,
    niter_dykstra=500,
    alpha=None,
    gamma=None,
    verbose=False,
    log=None,
    init=None,
    U=None,
    V=None,
):
    """Solve the unbalanced optimal transport problem with source using the Douglas-\
        Rachford algorithm.

    Given the source and destination densities rho0 and rho1, the cost matrix T, the \
        length scales ll,
    the constraint function H, and the right-hand side F, this function computes the \
        geodesic for the
    unbalanced optimal transport problem with source using the Douglas-Rachford algorithm.

    Args:
        rho0 (array): The source density.
        rho1 (array): The destination density.
        T (array): The cost matrix.
        ll (tuple): The length scales of the domain.
        H (array): The constraint function. If None, the algorithm will solve the \
            standard optimal transport problem.
        F (array): The right-hand side of the constraint. If None, the algorithm will \
            solve the standard optimal transport problem.
        p (float): The p-norm for the energy functional.
        q (float): The q-norm for the energy functional.
        delta (float): The scaling factor for the grid.
        niter (int): The number of iterations for the algorithm.
        dykstra (bool): If True, use the Dykstra algorithm to enforce the positivity constraint.
        eps (float): The tolerance for the Dykstra algorithm.
        alpha (float): The step size for the Douglas-Rachford algorithm.
        gamma (float): The regularization parameter for the Douglas-Rachford algorithm.
        verbose (bool): If True, print the progress of the algorithm.
        log (dict): The dictionary to store norms of pre_lambda, lambda, and the first \
            order condition. Assumed to have the keys 'pre_lambda', 'lambda', and \
            'first_order_condition' and the values are lists to store the norms.
        init (str) : The initialization method for the algorithm. If None, the algorithm \
            will use linear interpolation. If 'fisher-rao', the algorithm will use the \
            Fisher-Rao geodesic as the initialization. If 'manual', the algorithm will \
            use the provided U and V as the initialization.
        U (array): The initial value for U if init='manual'.
        V (array): The initial value for V if init='manual'.

    Returns:
        z (CSvar): The optimal transport solution.
        (Flist, Clist, HFlist) (tuple): The list of energy, distance from the \
            continuity equation, and distance from the constraint function at each \
            iteration. If H and F are None, HFlist is None.

    """
    assert delta > 0, "Delta must be positive"
    source = q >= 1.0  # Check if source problem

    nx = get_backend_ext(rho0, rho1)

    def prox1(y: grids.CSvar, x: grids.CSvar, source, gamma, p, q):
        projCE_(y.U, x.U, rho0 * delta**rho0.ndim, rho1 * delta**rho0.ndim, source)
        proxF_(y.V, x.V, gamma, p, q)

    def prox2(
        y,
        x,
        Q,
        solve=None,
        HQH=None,
        H=None,
        F=None,
        dykstra=False,
        big_matrix=False,
        eps=10e-3,
        niter_dykstra=1000,
    ):
        if HQH is None or H is None or F is None:
            projinterp_(y, x, Q, log)
        else:
            if big_matrix:
                projinterp_constraint_big_matrix(y, x, solve, H, F, log)
                return
            if dykstra:
                projinterp_constraint_dykstra(
                    y, x, Q, HQH, H, F, eps=10e-3, maxiter=niter_dykstra, log=log
                )
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
    w, x, y, z = [grids.CSvar(rho0, rho1, T, ll, init, U, V) for _ in range(4)]

    # Change of variable for scale adjustment
    for var in [w, x, y, z]:
        var.dilate_grid(1 / delta)
        var.rho1 *= delta**rho0.ndim
        var.rho0 *= delta**rho0.ndim

    # Precompute projection interpolation operators if needed
    Q = precomputeProjInterp(x.cs, rho0, rho1)
    HQH = precomputeHQH(Q[0], H, x.cs, x.ll) if H is not None else None
    PPT = precomputePPT(H, list(x.cs), x.ll) if H is not None else None
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
            lambda y, x: prox1(y, x, source, gamma, p, q),
            lambda y, x: prox2(
                y, x, Q, solve, HQH, H, F, dykstra, big_matrix, eps, niter_dykstra
            ),
            alpha,
        )

        Flist[i] = z.energy(delta, p, q)
        Clist[i] = z.dist_from_CE()
        Ilist[i] = z.dist_from_interp()
        if H is not None:
            HFlist[i] = z.dist_from_constraint(H, F)

    # Final projection and positive density adjustment
    projCE_(z.U, z.U, rho0 * delta**rho0.ndim, rho1 * delta**rho0.ndim, source)
    z.proj_positive()
    z.dilate_grid(delta)  # Adjust back to original scale
    z.interp_()  # Final interpolation adjustment

    if verbose:
        print("\nDone.")

    return z, (Flist, Clist, Ilist, HFlist)
