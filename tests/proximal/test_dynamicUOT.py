import proximal.backend_extension as be
import proximal.dynamicUOT as dyn
import proximal.grids as gr
import numpy as np
import scipy as sp
import torch
import pytest


class TestRoot:
    def test_root_numpy_positive_delta(self):
        coeff = np.array([1, -4, 6, -24]).astype(np.float32)
        nx = be.get_backend_ext(coeff)
        assert np.allclose(dyn.root(*coeff, nx), np.array([4.0]))

    def test_root_torch_positive_delta(self):
        coeff = torch.tensor([1, -4, 6, -24]).float()
        nx = be.get_backend_ext(coeff)
        assert np.allclose(dyn.root(*coeff, nx), torch.tensor([4.0]))

    def test_root_numpy_negative_delta(self):
        coeff = np.array([1, -5, 1, -5]).astype(np.float32)
        nx = be.get_backend_ext(coeff)
        assert np.allclose(dyn.root(*coeff, nx), np.array([5.0]))

    def test_root_torch_negative_delta(self):
        coeff = torch.tensor([1, -5, 1, -5]).float()
        nx = be.get_backend_ext(coeff)
        assert np.allclose(dyn.root(*coeff, nx), torch.tensor([5.0]))

    def test_root_numpy_zero_delta(self):
        coeff = np.array([1, -3, 3, -1]).astype(np.float32)
        nx = be.get_backend_ext(coeff)
        assert np.allclose(dyn.root(*coeff, nx), np.array([1.0]))

    def test_root_torch_zero_delta(self):
        coeff = torch.tensor([1, -3, 3, -1]).float()
        nx = be.get_backend_ext(coeff)
        assert np.allclose(dyn.root(*coeff, nx), torch.tensor([1.0]))

    def test_root_numpy_multi_input(self):
        a = np.array([1, 1, 1]).astype(np.float32)
        b = np.array([-4, -5, -3]).astype(np.float32)
        c = np.array([6, 1, 3]).astype(np.float32)
        d = np.array([-24, -5, -1]).astype(np.float32)
        nx = be.get_backend_ext(a, b, c, d)
        assert np.allclose(dyn.root(a, b, c, d, nx), np.array([4.0, 5.0, 1.0]))


class TestProxB:
    def test_proxB_numpy_small(self):
        R = np.array([[1, 2], [3, 4]]).astype(np.float64)
        M = np.array([[5, 6], [7, 8]]).astype(np.float64)
        Z = np.array([[9, 10], [11, 12]]).astype(np.float64)
        M = [M, Z]
        nx = be.get_backend_ext(R, M[0], M[1])
        gamma = 1.0
        destR = np.zeros_like(R)
        destM = [np.zeros_like(M[0]), np.zeros_like(M[1])]
        dyn.proxB_(destR, destM, R, M, gamma=gamma, nx=nx)
        assert np.allclose(
            destR,
            np.array(
                [
                    [3.55474548856287, 4.36366412618957],
                    [5.20656376605373, 6.07669399212356],
                ]
            ),
        )
        assert np.allclose(
            destM[0],
            np.array(
                [
                    [3.90224382184357, 4.88136172235257],
                    [5.87216175264550, 6.86952862326616],
                ]
            ),
        )
        assert np.allclose(
            destM[1],
            np.array(
                [
                    [7.02403887931843, 8.13560287058762],
                    [9.22768275415721, 10.3042929348992],
                ]
            ),
        )

    def test_proxB_torch_small(self):
        R = torch.tensor([[1, 2], [3, 4]]).float()
        M = torch.tensor([[5, 6], [7, 8]]).float()
        Z = torch.tensor([[9, 10], [11, 12]]).float()
        M = [M, Z]
        nx = be.get_backend_ext(R, M[0], M[1])
        gamma = 1.0
        destR = torch.zeros_like(R)
        destM = [torch.zeros_like(M[0]), torch.zeros_like(M[1])]
        dyn.proxB_(destR, destM, R, M, gamma=gamma, nx=nx)
        assert torch.allclose(
            destR,
            torch.tensor(
                [
                    [3.55474548856287, 4.36366412618957],
                    [5.20656376605373, 6.07669399212356],
                ]
            ),
        )
        assert torch.allclose(
            destM[0],
            torch.tensor(
                [
                    [3.90224382184357, 4.88136172235257],
                    [5.87216175264550, 6.86952862326616],
                ]
            ),
        )
        assert torch.allclose(
            destM[1],
            torch.tensor(
                [
                    [7.02403887931843, 8.13560287058762],
                    [9.22768275415721, 10.3042929348992],
                ]
            ),
        )


class TestPoisson:
    def test_poisson_numpy_nosource_zero(self):
        f = np.zeros((3, 3))
        nx = be.get_backend_ext(f)
        dyn.poisson_(f, (1.0, 1.0), False, nx)
        assert np.allclose(f, np.zeros((3, 3)))

    def test_poisson_torch_nosource_zero(self):
        f = torch.zeros((3, 3))
        nx = be.get_backend_ext(f)
        dyn.poisson_(f, (1.0, 1.0), False, nx)
        assert torch.allclose(f, torch.zeros((3, 3)))

    def test_poisson_numpy_source_zero(self):
        f = np.zeros((3, 3))
        nx = be.get_backend_ext(f)
        dyn.poisson_(f, (1.0, 1.0), True, nx)
        assert np.allclose(f, np.zeros((3, 3)))

    def test_poisson_torch_source_zero(self):
        f = torch.zeros((3, 3))
        nx = be.get_backend_ext(f)
        dyn.poisson_(f, (1.0, 1.0), True, nx)
        assert torch.allclose(f, torch.zeros((3, 3)))

    def test_poisson_numpy_nosource_nonzero(self):
        N_0 = 100
        N_1 = 100
        tol = (1 / N_0) ** 2 + (1 / N_1) ** 2
        ll = (1.0, 1.0)
        t = ll[0] * (np.arange(N_0) + 0.5) / N_0
        x = ll[1] * (np.arange(N_1) + 0.5) / N_1
        T, X = np.meshgrid(t, x)
        f = 8 * np.pi**2 * np.cos(np.pi * (2 * T - 1)) * np.cos(np.pi * (2 * X - 1))
        nx = be.get_backend_ext(f)
        dyn.poisson_(f, ll, False, nx)
        np.testing.assert_allclose(
            f,
            np.cos(np.pi * (2 * T - 1)) * np.cos(np.pi * (2 * X - 1)),
            atol=tol,
            rtol=tol,
        )

    def test_poisson_torch_nosource_nonzero(self):
        N_0 = 100
        N_1 = 100
        tol = (1 / N_0) ** 2 + (1 / N_1) ** 2
        ll = (1.0, 1.0)
        t = ll[0] * (torch.arange(N_0) + 0.5) / N_0
        x = ll[1] * (torch.arange(N_1) + 0.5) / N_1
        T, X = torch.meshgrid(t, x)
        f = (
            8
            * np.pi**2
            * torch.cos(np.pi * (2 * T - 1))
            * torch.cos(np.pi * (2 * X - 1))
        )
        nx = be.get_backend_ext(f)
        dyn.poisson_(f, ll, False, nx)
        torch.testing.assert_allclose(
            f,
            torch.cos(np.pi * (2 * T - 1)) * torch.cos(np.pi * (2 * X - 1)),
            atol=tol,
            rtol=tol,
        )

    def test_posison_numpy_source_nonzero(self):
        N_0 = 100
        N_1 = 100
        tol = (1 / N_0) ** 2 + (1 / N_1) ** 2
        ll = (1.0, 1.0)
        t = ll[0] * (np.arange(N_0) + 0.5) / N_0
        x = ll[1] * (np.arange(N_1) + 0.5) / N_1
        T, X = np.meshgrid(t, x)
        f = (
            (8 * np.pi**2 + 1)
            * np.cos(np.pi * (2 * T - 1))
            * np.cos(np.pi * (2 * X - 1))
        )
        nx = be.get_backend_ext(f)
        dyn.poisson_(f, ll, True, nx)
        np.testing.assert_allclose(
            f,
            np.cos(np.pi * (2 * T - 1)) * np.cos(np.pi * (2 * X - 1)),
            atol=tol,
            rtol=tol,
        )


class TestPoissonMixedBC:
    def build_test_numpy(self, Nt=8, Nx=16, kt=2, kx=3, Lt=1.0, Lx=1.0, source=False):
        ht, hx = Lt / Nt, Lx / Nx
        j = np.arange(Nt)[:, None]
        i = np.arange(Nx)[None, :]

        u_exact = np.cos(np.pi * kt * (j + 0.5) / Nt) * np.cos(2 * np.pi * kx * i / Nx)
        lam = (2 * np.cos(np.pi * kt / Nt) - 2) / ht**2 + (
            2 * np.cos(2 * np.pi * kx / Nx) - 2
        ) / hx**2

        if source:
            f = -(lam - 1.0) * u_exact
        else:
            f = -lam * u_exact
        return u_exact.copy(), f  # copy so we can overwrite f in-place

    def test_poisson_numpy_mixedbc(self):
        Nt = 8
        Nx = 16
        kt = 2
        kx = 3
        Lt = 1.0
        Lx = 1.0
        source = False

        u_exact, f = self.build_test_numpy(Nt, Nx, kt, kx, Lt, Lx, source)
        nx = be.get_backend_ext(f)
        dyn.poisson_mixed_bc_(f, (Lt, Lx), source, nx)
        np.testing.assert_allclose(f, u_exact, rtol=0, atol=1e-12)


class TestMinusInterior:
    def test_minus_interior_numpy_small(self):
        M = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [6, 7, 8, 9], [10, 11, 12, 13]]
        ).astype(np.float64)
        dest = M.copy()
        dpk = np.array([[6, 6, 7, 8], [6, 7, 8, 9]]).astype(np.float64)
        cs = (3, 4)
        dim = 0
        dyn.minus_interior_(dest, M, dpk, cs, dim)
        assert np.allclose(
            dest,
            np.array([[1, 2, 3, 4], [-1, 0, 0, 0], [0, 0, 0, 0], [10, 11, 12, 13]]),
        )

    def test_minus_interior_torch_small(self):
        M = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8], [6, 7, 8, 9], [10, 11, 12, 13]]
        ).float()
        dest = M.clone()
        dpk = torch.tensor([[6, 6, 7, 8], [6, 7, 8, 9]]).float()
        cs = (3, 4)
        dim = 0
        dyn.minus_interior_(dest, M, dpk, cs, dim)
        assert torch.allclose(
            dest,
            torch.tensor(
                [[1, 2, 3, 4], [-1, 0, 0, 0], [0, 0, 0, 0], [10, 11, 12, 13]],
                dtype=torch.float32,
            ),
        )


class TestProjCE:
    def test_projCE_numpy_small(self):
        rho0 = np.array([1, 2, 3, 4]).astype(np.float32)
        rho1 = np.array([5, 6, 7, 8]).astype(np.float32)
        T = 5
        cs = (T, 4)
        ll = (1.0, 1.0)
        shapes_staggered = gr.get_staggered_shape(cs)
        D = [np.zeros(shapes_staggered[k]) for k in range(2)]
        D[0] = gr.linear_interpolation(rho0, rho1, cs[0])
        U = gr.Svar(cs, ll, D, Z=np.zeros(cs))
        dyn.projCE_(U, U, rho0, rho1, True)
        np.testing.assert_allclose(U.remainder_CE(), 0, atol=1e-5, rtol=1e-8)

    def test_projCE_torch_small(self):
        rho0 = torch.tensor([1, 2, 3, 4]).float()
        rho1 = torch.tensor([5, 6, 7, 8]).float()
        T = 5
        cs = (T, 4)
        ll = (1.0, 1.0)
        shapes_staggered = gr.get_staggered_shape(cs)
        D = [torch.zeros(shapes_staggered[k]) for k in range(2)]
        D[0] = gr.linear_interpolation(rho0, rho1, cs[0])
        U = gr.Svar(cs, ll, D, Z=torch.zeros(cs))
        dyn.projCE_(U, U, rho0, rho1, True)
        torch.testing.assert_close(
            U.remainder_CE(), torch.zeros(cs), atol=1e-5, rtol=1e-8
        )

    def test_projCE_numpy_small2(self):
        rho0 = np.array([1, 2, 1]).astype(np.float32)
        rho1 = np.array([4, 5, 3]).astype(np.float32)
        T = 2
        cs = (T, 3)
        ll = (1.0, 1.0)
        shapes_staggered = gr.get_staggered_shape(cs)
        D = [np.zeros(shapes_staggered[k]) for k in range(2)]
        D[0] = gr.linear_interpolation(rho0, rho1, cs[0])
        U = gr.Svar(cs, ll, D, Z=np.zeros(cs))
        dyn.projCE_(U, U, rho0, rho1, source=True)
        np.testing.assert_allclose(U.remainder_CE(), 0, atol=1e-5, rtol=1e-8)

    def test_proJCE_numpy_large(self):
        rho0 = np.random.rand(256).astype(np.float32)
        rho1 = np.random.rand(256).astype(np.float32)
        T = 100
        cs = (T, 256)
        ll = (1.0, 1.0)
        shapes_staggered = gr.get_staggered_shape(cs)
        D = [np.zeros(shapes_staggered[k]) for k in range(2)]
        D[0] = gr.linear_interpolation(rho0, rho1, cs[0])
        U = gr.Svar(cs, ll, D, Z=np.zeros(cs))
        dyn.projCE_(U, U, rho0, rho1, True)
        np.testing.assert_allclose(U.remainder_CE(), 0, atol=1e-5, rtol=1e-8)

    """ projCE with torch seems to have issues but I will ignore it for now
    def test_projCE_torch_large(self):
        rho0 = torch.rand(256).float()
        rho1 = torch.rand(256).float()
        T = 100
        cs = (T, 256)
        ll = (1.0, 1.0)
        shapes_staggered = gr.get_staggered_shape(cs)
        D = [torch.zeros(shapes_staggered[k]) for k in range(2)]
        D[0] = gr.linear_interpolation(rho0, rho1, cs[0])
        U = gr.Svar(cs, ll, D, Z=torch.zeros(cs))
        dyn.projCE_(U, U, rho0, rho1, True)
        torch.testing.assert_close(
            U.remainder_CE(), torch.zeros(cs), atol=1e-5, rtol=1e-8
        )
    """

    def test_projCE_numpy_small_periodic(self):
        rho0 = np.array([1, 2, 3, 4]).astype(np.float32)
        rho1 = np.array([5, 6, 7, 8]).astype(np.float32)
        T = 5
        cs = (T, 4)
        ll = (1.0, 1.0)
        shapes_staggered = gr.get_staggered_shape(cs, periodic=True)
        D = [np.zeros(shapes_staggered[k]) for k in range(2)]
        D[0] = gr.linear_interpolation(rho0, rho1, cs[0])
        U = gr.Svar(cs, ll, D, Z=np.zeros(cs))
        dyn.projCE_(U, U, rho0, rho1, source=True, periodic=True)
        np.testing.assert_allclose(
            U.remainder_CE(periodic=True), 0, atol=1e-5, rtol=1e-8
        )

    def test_projCE_numpy_small_periodic2(self):
        rho0 = np.array([1, 2, 1]).astype(np.float32)
        rho1 = np.array([4, 5, 3]).astype(np.float32)
        T = 2
        cs = (T, 3)
        ll = (1.0, 1.0)
        shapes_staggered = gr.get_staggered_shape(cs, periodic=True)
        D = [np.zeros(shapes_staggered[k]) for k in range(2)]
        D[0] = gr.linear_interpolation(rho0, rho1, cs[0])
        np.testing.assert_allclose(D[0], [[1, 2, 1], [2.5, 3.5, 2], [4, 5, 3]])
        U = gr.Svar(cs, ll, D, Z=np.zeros(cs))
        dyn.projCE_(U, U, rho0, rho1, source=True, periodic=True)
        np.testing.assert_allclose(
            U.remainder_CE(periodic=True), 0, atol=1e-5, rtol=1e-8
        )

    def test_projCE_torch_small_periodic(self):
        rho0 = torch.tensor([1, 2, 3, 4]).float()
        rho1 = torch.tensor([5, 6, 7, 8]).float()
        T = 5
        cs = (T, 4)
        ll = (1.0, 1.0)
        shapes_staggered = gr.get_staggered_shape(cs, periodic=True)
        D = [torch.zeros(shapes_staggered[k]) for k in range(2)]
        D[0] = gr.linear_interpolation(rho0, rho1, cs[0])
        U = gr.Svar(cs, ll, D, Z=torch.zeros(cs))
        dyn.projCE_(U, U, rho0, rho1, source=True, periodic=True)
        torch.testing.assert_close(
            U.remainder_CE(periodic=True), torch.zeros(cs), atol=1e-5, rtol=1e-8
        )

    def test_projCE_numpy_large_periodic(self):
        T = 100
        N = 256
        rho0 = np.random.rand(N).astype(np.float32)
        rho1 = np.random.rand(N).astype(np.float32)
        cs = (T, N)
        ll = (1.0, 1.0)
        shapes_staggered = gr.get_staggered_shape(cs, periodic=True)
        D = [np.zeros(shapes_staggered[k]) for k in range(2)]
        D[0] = gr.linear_interpolation(rho0, rho1, cs[0])
        U = gr.Svar(cs, ll, D, Z=np.zeros(cs))
        dyn.projCE_(U, U, rho0, rho1, source=True, periodic=True)
        np.testing.assert_allclose(
            U.remainder_CE(periodic=True), 0, atol=1e-5, rtol=1e-8
        )

    """
    def test_projCE_torch_large_periodic(self):
        T = 200
        N = 10
        rho0 = torch.tensor([4, 3, 1, 3, 6, 2, 1, 5, 6, 1]).float()
        rho1 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).float()
        cs = (T, N)
        ll = (1.0, 1.0)
        shapes_staggered = gr.get_staggered_shape(cs, periodic=True)
        D = [torch.zeros(shapes_staggered[k]) for k in range(2)]
        D[0] = gr.linear_interpolation(rho0, rho1, cs[0])
        U = gr.Svar(cs, ll, D, Z=torch.zeros(cs))
        dyn.projCE_(U, U, rho0, rho1, source=True, periodic=True)
        torch.testing.assert_close(
            U.remainder_CE(periodic=True), torch.zeros(cs), atol=1e-5, rtol=1e-8
        )
    """


class TestInvQ_Mul_A_:
    def test_invQ_Mul_A_numpy_small1(self):
        src = np.array([[1, 2], [3, 4]]).astype(np.float32)
        Q = np.array([[1, 2], [0, 1]]).astype(np.float32)
        nx = be.get_backend_ext(src, Q)
        dyn.invQ_mul_A_(src, Q, src, 0, nx)
        np.testing.assert_allclose(src, np.array([[-5, -6], [3, 4]]))

    def test_invQ_Mul_A_torch_small1(self):
        src = torch.tensor([[1, 2], [3, 4]]).float()
        Q = torch.tensor([[1, 2], [0, 1]]).float()
        nx = be.get_backend_ext(src, Q)
        dyn.invQ_mul_A_(src, Q, src, 0, nx)
        torch.testing.assert_close(src, torch.tensor([[-5, -6], [3, 4]]).float())

    def test_invQ_Mul_A_numpy_small2(self):
        src = np.array([[1, 2], [3, 4]]).astype(np.float32)
        Q = np.array([[1, 2], [0, 1]]).astype(np.float32)
        nx = be.get_backend_ext(src, Q)
        dyn.invQ_mul_A_(src, Q, src, 1, nx)
        np.testing.assert_allclose(src, np.array([[-3, 2], [-5, 4]]))

    def test_invQ_Mul_A_torch_small2(self):
        src = torch.tensor([[1, 2], [3, 4]]).float()
        Q = torch.tensor([[1, 2], [0, 1]]).float()
        nx = be.get_backend_ext(src, Q)
        dyn.invQ_mul_A_(src, Q, src, 1, nx)
        torch.testing.assert_close(src, torch.tensor([[-3, 2], [-5, 4]]).float())


class TestProjInterp_:
    def test_projInterp_numpy_small(self):
        rho0 = np.array([1, 2]).astype(np.float32)
        rho1 = np.array([5, 6]).astype(np.float32)
        T = 2
        ll = (1.0, 1.0)
        x = gr.CSvar(rho0, rho1, T, ll)
        y = gr.CSvar(rho0, rho1, T, ll)
        x.U.D[1] = np.array([[0, 2, 0], [2, 4, 0]]).astype(np.float32)
        x.U.Z = np.array([[0, 0], [0, 0]]).astype(np.float32)
        x.interp_()

        assert x.U.N == 2

        np.testing.assert_allclose(
            x.U.D[0], np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
        )
        np.testing.assert_allclose(
            x.V.D[0], np.array([[2, 3], [4, 5]]).astype(np.float32)
        )
        np.testing.assert_allclose(
            x.V.D[1], np.array([[1, 1], [3, 2]]).astype(np.float32)
        )

        identities = [
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32),
        ]
        Q = [
            np.array([[1, 2, 3], [0, 1, 5], [0, 0, 1]]).astype(np.float32),
            np.array([[1, 7, 8], [0, 1, 9], [0, 0, 1]]).astype(np.float32),
        ]
        dyn.projinterp_(y, x, identities)
        dyn.projinterp_(x, x, Q)
        np.testing.assert_allclose(y.U.D[0], np.array([[2, 3.5], [6, 8], [7, 8.5]]))
        np.testing.assert_allclose(y.U.D[1], np.array([[0.5, 3, 0.5], [3.5, 6.5, 1.0]]))
        np.testing.assert_allclose(
            x.U.D[0], np.array([[39, 47], [-29, -34.5], [7, 8.5]])
        )
        np.testing.assert_allclose(x.U.D[1], np.array([[7, -1.5, 0.5], [13, -2.5, 1]]))
        np.testing.assert_allclose(x.U.Z, np.array([[0, 0], [0, 0]]))


class TestProjInterp_Constraint:
    def test_projInterp_constraint_numpy_small(self):
        rho0 = np.array([1, 2]).astype(np.float32)
        rho1 = np.array([5, 6]).astype(np.float32)
        T = 2
        cs = (2, 2)
        ll = (1.0, 1.0)
        x = gr.CSvar(rho0, rho1, T, ll)
        y = gr.CSvar(rho0, rho1, T, ll)
        x.U.D[1] = np.array([[0, 2, 0], [2, 4, 0]]).astype(np.float32)
        x.interp_()
        identities = [
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32),
        ]
        H = np.ones(cs, dtype=np.float32)
        F = np.array([7, 8], dtype=np.float32)
        dyn.projinterp_constraint_(y, x, identities, np.eye(2), H, F)
        np.testing.assert_allclose(
            y.U.D[0],
            np.array([[2.53125, 4.03125], [6.6875, 8.6875], [7.15625, 8.65625]]),
        )
        np.testing.assert_allclose(y.U.D[1], np.array([[0.5, 3, 0.5], [3.5, 6.5, 1.0]]))

    def test_projInterp_constraint_torch_small(self):
        rho0 = torch.tensor([1, 2]).float()
        rho1 = torch.tensor([5, 6]).float()
        T = 2
        cs = (2, 2)
        ll = (1.0, 1.0)
        x = gr.CSvar(rho0, rho1, T, ll)
        y = gr.CSvar(rho0, rho1, T, ll)
        x.U.D[1] = torch.tensor([[0, 2, 0], [2, 4, 0]]).float()
        # x.U.Z = torch.tensor([[0, 0], [0, 0]]).float()
        x.interp_()
        identities = [
            torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float(),
            torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float(),
        ]
        H = torch.ones(cs, dtype=torch.float32)
        F = torch.tensor([7, 8], dtype=torch.float32)
        dyn.projinterp_constraint_(y, x, identities, torch.eye(2), H, F)
        torch.testing.assert_close(
            y.U.D[0],
            torch.tensor([[2.53125, 4.03125], [6.6875, 8.6875], [7.15625, 8.65625]]),
        )
        torch.testing.assert_close(
            y.U.D[1], torch.tensor([[0.5, 3, 0.5], [3.5, 6.5, 1.0]])
        )

    def test_projInterp_constraint_numpy_small_2(self):
        rho0 = np.array([1, 2]).astype(np.float32)
        rho1 = np.array([5, 6]).astype(np.float32)
        T = 2
        cs = (2, 2)
        ll = (1.0, 1.0)
        x = gr.CSvar(rho0, rho1, T, ll)
        x.U.D[1] = np.array([[0, 2, 0], [2, 4, 0]]).astype(np.float32)
        x.interp_()
        Q = [
            np.array([[1, 2, 3], [0, 1, 5], [0, 0, 1]]).astype(np.float32),
            np.array([[1, 7, 8], [0, 1, 9], [0, 0, 1]]).astype(np.float32),
        ]
        H = np.ones(cs, dtype=np.float32)
        F = np.array([7, 8], dtype=np.float32)
        dyn.projinterp_constraint_(x, x, Q, np.array([[1, 3], [0, 1]]), H, F)
        np.testing.assert_allclose(
            x.U.D[0],
            np.array([[78.65625, 86.65625], [-63.65625, -69.15625], [12.0, 13.5]]),
        )
        np.testing.assert_allclose(x.U.D[1], np.array([[7, -1.5, 0.5], [13, -2.5, 1]]))

    def test_projInterp_constraint_torch_small_2(self):
        rho0 = torch.tensor([1, 2]).float()
        rho1 = torch.tensor([5, 6]).float()
        T = 2
        cs = (2, 2)
        ll = (1.0, 1.0)
        x = gr.CSvar(rho0, rho1, T, ll)
        x.U.D[1] = torch.tensor([[0, 2, 0], [2, 4, 0]]).float()
        x.interp_()
        Q = [
            torch.tensor([[1, 2, 3], [0, 1, 5], [0, 0, 1]]).float(),
            torch.tensor([[1, 7, 8], [0, 1, 9], [0, 0, 1]]).float(),
        ]
        H = torch.ones(cs, dtype=torch.float32)
        F = torch.tensor([7, 8], dtype=torch.float32)
        dyn.projinterp_constraint_(
            x, x, Q, torch.tensor([[1, 3], [0, 1]]).float(), H, F
        )
        torch.testing.assert_close(
            x.U.D[0],
            torch.tensor([[78.65625, 86.65625], [-63.65625, -69.15625], [12.0, 13.5]]),
        )
        torch.testing.assert_close(
            x.U.D[1], torch.tensor([[7, -1.5, 0.5], [13, -2.5, 1]])
        )


class TestPrecomputeProjectInterp:
    def test_precompute_project_interp_numpy_small(self):
        cs = (3, 4)
        z = np.zeros(cs, dtype=np.float64)
        Q = dyn.precomputeProjInterp(cs, z, z)
        np.testing.assert_allclose(
            Q[0],
            np.array(
                [
                    [5.0 / 4.0, 1.0 / 4.0, 0.0, 0.0],
                    [1.0 / 4.0, 6.0 / 4.0, 1.0 / 4.0, 0.0],
                    [0.0, 1.0 / 4.0, 6.0 / 4.0, 1.0 / 4.0],
                    [0.0, 0.0, 1.0 / 4.0, 5.0 / 4.0],
                ]
            ),
        )
        np.testing.assert_allclose(
            Q[1],
            np.array(
                [
                    [5.0 / 4.0, 1.0 / 4.0, 0.0, 0.0, 0.0],
                    [1.0 / 4.0, 6.0 / 4.0, 1.0 / 4.0, 0.0, 0.0],
                    [0.0, 1.0 / 4.0, 6.0 / 4.0, 1.0 / 4.0, 0.0],
                    [0.0, 0.0, 1.0 / 4.0, 6.0 / 4.0, 1.0 / 4.0],
                    [0.0, 0.0, 0.0, 1.0 / 4.0, 5.0 / 4.0],
                ]
            ),
        )

    def test_precompute_project_interp_torch_small(self):
        cs = (3, 4)
        z = torch.zeros(cs, dtype=torch.float64)
        Q = dyn.precomputeProjInterp(cs, z, z)
        torch.testing.assert_close(
            Q[0],
            torch.tensor(
                [
                    [5.0 / 4.0, 1.0 / 4.0, 0.0, 0.0],
                    [1.0 / 4.0, 6.0 / 4.0, 1.0 / 4.0, 0.0],
                    [0.0, 1.0 / 4.0, 6.0 / 4.0, 1.0 / 4.0],
                    [0.0, 0.0, 1.0 / 4.0, 5.0 / 4.0],
                ],
                dtype=torch.float64,
            ),
        )
        torch.testing.assert_close(
            Q[1],
            torch.tensor(
                [
                    [5.0 / 4.0, 1.0 / 4.0, 0.0, 0.0, 0.0],
                    [1.0 / 4.0, 6.0 / 4.0, 1.0 / 4.0, 0.0, 0.0],
                    [0.0, 1.0 / 4.0, 6.0 / 4.0, 1.0 / 4.0, 0.0],
                    [0.0, 0.0, 1.0 / 4.0, 6.0 / 4.0, 1.0 / 4.0],
                    [0.0, 0.0, 0.0, 1.0 / 4.0, 5.0 / 4.0],
                ],
                dtype=torch.float64,
            ),
        )

    def test_precompute_project_interp_numpy_periodic(self):
        cs = (3, 4)
        z = np.zeros(cs, dtype=np.float64)
        Q = dyn.precomputeProjInterp(cs, z, z, periodic=True)
        np.testing.assert_allclose(
            Q[0],
            np.array(
                [
                    [5.0 / 4.0, 1.0 / 4.0, 0.0, 0.0],
                    [1.0 / 4.0, 6.0 / 4.0, 1.0 / 4.0, 0.0],
                    [0.0, 1.0 / 4.0, 6.0 / 4.0, 1.0 / 4.0],
                    [0.0, 0.0, 1.0 / 4.0, 5.0 / 4.0],
                ]
            ),
        )
        np.testing.assert_allclose(
            Q[1],
            np.array(
                [
                    [1.5, 0.25, 0.0, 0.25],
                    [0.25, 1.5, 0.25, 0.0],
                    [0.0, 0.25, 1.5, 0.25],
                    [0.25, 0.0, 0.25, 1.5],
                ]
            ),
        )

    def test_precompute_project_interp_torch_periodic(self):
        cs = (3, 4)
        z = torch.zeros(cs, dtype=torch.float64)
        Q = dyn.precomputeProjInterp(cs, z, z, periodic=True)
        torch.testing.assert_close(
            Q[0],
            torch.tensor(
                [
                    [5.0 / 4.0, 1.0 / 4.0, 0.0, 0.0],
                    [1.0 / 4.0, 6.0 / 4.0, 1.0 / 4.0, 0.0],
                    [0.0, 1.0 / 4.0, 6.0 / 4.0, 1.0 / 4.0],
                    [0.0, 0.0, 1.0 / 4.0, 5.0 / 4.0],
                ],
                dtype=torch.float64,
            ),
        )
        torch.testing.assert_close(
            Q[1],
            torch.tensor(
                [
                    [1.5, 0.25, 0.0, 0.25],
                    [0.25, 1.5, 0.25, 0.0],
                    [0.0, 0.25, 1.5, 0.25],
                    [0.25, 0.0, 0.25, 1.5],
                ],
                dtype=torch.float64,
            ),
        )


class TestprecomputeHQH:
    def test_precompute_HQH_numpy_small(self):
        cs = (3, 4)
        ll = (1.0, 1.0)
        z = np.zeros(cs, dtype=np.float64)
        Q = dyn.precomputeProjInterp(cs, z, z)
        H = np.ones(cs, dtype=np.float64)
        Q = dyn.precomputeHQH(Q[0], H, cs, ll)
        np.testing.assert_allclose(
            Q[:, 0],
            np.array([0.0784313725490196, 0.0294117647058824, -0.0049019607843137]),
        )

    def test_precompute_HQH_torch_small(self):
        cs = (3, 4)
        ll = (1.0, 1.0)
        z = torch.zeros(cs)
        Q = dyn.precomputeProjInterp(cs, z, z)
        H = torch.ones(cs)
        Q = dyn.precomputeHQH(Q[0], H, cs, ll)
        torch.testing.assert_close(
            Q[:, 0],
            torch.tensor([0.0784313725490196, 0.0294117647058824, -0.0049019607843137]),
        )


class TestFlattenArray:
    def test_flatten_array_2D_numpy_small(self):
        a = np.array([[1, 2], [3, 4]]).astype(np.float32)
        fastest_axis = 0
        b = dyn.flatten_array(a, fastest_axis)
        np.testing.assert_allclose(b, np.array([1, 3, 2, 4]).astype(np.float32))

    def test_flatten_array_2D_torch_small(self):
        a = torch.tensor([[1, 2], [3, 4]]).float()
        fastest_axis = 0
        b = dyn.flatten_array(a, fastest_axis)
        torch.testing.assert_close(b, torch.tensor([1, 3, 2, 4]).float())

    def test_flatten_array_2D_numpy_small_axis1(self):
        a = np.array([[1, 2], [3, 4]]).astype(np.float32)
        fastest_axis = 1
        b = dyn.flatten_array(a, fastest_axis)
        np.testing.assert_allclose(b, np.array([1, 2, 3, 4]).astype(np.float32))

    def test_flatten_array_2D_torch_small_axis1(self):
        a = torch.tensor([[1, 2], [3, 4]]).float()
        fastest_axis = 1
        b = dyn.flatten_array(a, fastest_axis)
        torch.testing.assert_close(b, torch.tensor([1, 2, 3, 4]).float())

    def test_flatten_array_3D_numpy_small(self):
        a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.float32)
        fastest_axis = 0
        b = dyn.flatten_array(a, fastest_axis)
        np.testing.assert_allclose(
            b, np.array([1, 5, 3, 7, 2, 6, 4, 8]).astype(np.float32)
        )

    def test_flatten_array_3D_torch_small(self):
        a = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).float()
        fastest_axis = 0
        b = dyn.flatten_array(a, fastest_axis)
        torch.testing.assert_close(b, torch.tensor([1, 5, 3, 7, 2, 6, 4, 8]).float())


class TestUnflattenArray:
    def test_unflatten_array_2D_numpy_small(self):
        a = np.array([1, 3, 2, 4]).astype(np.float32)
        shape = (2, 2)
        fastest_axis = 0
        b = dyn.unflatten_array(a, shape, fastest_axis)
        np.testing.assert_allclose(b, np.array([[1, 2], [3, 4]]).astype(np.float32))

    def test_unflatten_array_2D_torch_small(self):
        a = torch.tensor([1, 3, 2, 4]).float()
        shape = (2, 2)
        fastest_axis = 0
        b = dyn.unflatten_array(a, shape, fastest_axis)
        torch.testing.assert_close(b, torch.tensor([[1, 2], [3, 4]]).float())

    def test_unflatten_array_2D_numpy_small_axis1(self):
        a = np.array([1, 2, 3, 4]).astype(np.float32)
        shape = (2, 2)
        fastest_axis = 1
        b = dyn.unflatten_array(a, shape, fastest_axis)
        np.testing.assert_allclose(b, np.array([[1, 2], [3, 4]]).astype(np.float32))

    def test_unflatten_array_2D_torch_small_axis1(self):
        a = torch.tensor([1, 2, 3, 4]).float()
        shape = (2, 2)
        fastest_axis = 1
        b = dyn.unflatten_array(a, shape, fastest_axis)
        torch.testing.assert_close(b, torch.tensor([[1, 2], [3, 4]]).float())

    def test_unflatten_array_3D_numpy_small(self):
        a = np.array([1, 5, 3, 7, 2, 6, 4, 8]).astype(np.float32)
        shape = (2, 2, 2)
        fastest_axis = 0
        b = dyn.unflatten_array(a, shape, fastest_axis)
        np.testing.assert_allclose(
            b, np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.float32)
        )

    def test_unflatten_array_3D_torch_small(self):
        a = torch.tensor([1, 5, 3, 7, 2, 6, 4, 8]).float()
        shape = (2, 2, 2)
        fastest_axis = 0
        b = dyn.unflatten_array(a, shape, fastest_axis)
        torch.testing.assert_close(
            b, torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).float()
        )


class TestFlattenUnflatten:
    def test_flatten_unflatten_2D_numpy_random(self):
        a = np.random.rand(3, 4).astype(np.float32)
        fastest_axis = 0
        b = dyn.flatten_array(a, fastest_axis)
        c = dyn.unflatten_array(b, a.shape, fastest_axis)
        np.testing.assert_allclose(a, c)

    def test_flatten_unflatten_2D_torch_random(self):
        a = torch.rand(3, 4).float()
        fastest_axis = 0
        b = dyn.flatten_array(a, fastest_axis)
        c = dyn.unflatten_array(b, a.shape, fastest_axis)
        torch.testing.assert_close(a, c)

    def test_flatten_unflatten_3D_numpy_random(self):
        a = np.random.rand(3, 4, 5).astype(np.float32)
        fastest_axis = 0
        b = dyn.flatten_array(a, fastest_axis)
        c = dyn.unflatten_array(b, a.shape, fastest_axis)
        np.testing.assert_allclose(a, c)

    def test_flatten_unflatten_3D_torch_random(self):
        a = torch.rand(3, 4, 5).float()
        fastest_axis = 0
        b = dyn.flatten_array(a, fastest_axis)
        c = dyn.unflatten_array(b, a.shape, fastest_axis)
        torch.testing.assert_close(a, c)

    def test_flatten_unflatten_2D_numpy_random_axis1(self):
        a = np.random.rand(3, 4).astype(np.float32)
        fastest_axis = 1
        b = dyn.flatten_array(a, fastest_axis)
        c = dyn.unflatten_array(b, a.shape, fastest_axis)
        np.testing.assert_allclose(a, c)

    def test_flatten_unflatten_2D_torch_random_axis1(self):
        a = torch.rand(3, 4).float()
        fastest_axis = 1
        b = dyn.flatten_array(a, fastest_axis)
        c = dyn.unflatten_array(b, a.shape, fastest_axis)
        torch.testing.assert_close(a, c)

    def test_flatten_unflatten_3D_numpy_random_axis1(self):
        a = np.random.rand(3, 4, 5).astype(np.float32)
        fastest_axis = 1
        b = dyn.flatten_array(a, fastest_axis)
        c = dyn.unflatten_array(b, a.shape, fastest_axis)
        np.testing.assert_allclose(a, c)

    def test_flatten_unflatten_3D_torch_random_axis1(self):
        a = torch.rand(3, 4, 5).float()
        fastest_axis = 1
        b = dyn.flatten_array(a, fastest_axis)
        c = dyn.unflatten_array(b, a.shape, fastest_axis)
        torch.testing.assert_close(a, c)

    def test_flatten_unflatten_3D_numpy_random_axis2(self):
        a = np.random.rand(3, 4, 5).astype(np.float32)
        fastest_axis = 2
        b = dyn.flatten_array(a, fastest_axis)
        c = dyn.unflatten_array(b, a.shape, fastest_axis)
        np.testing.assert_allclose(a, c)

    def test_flatten_unflatten_3D_torch_random_axis2(self):
        a = torch.rand(3, 4, 5).float()
        fastest_axis = 2
        b = dyn.flatten_array(a, fastest_axis)
        c = dyn.unflatten_array(b, a.shape, fastest_axis)
        torch.testing.assert_close(a, c)


class TestVectorizeVF:
    def test_vectorize_VF_numpy_2D(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        D = [np.zeros(cs) for _ in range(2)]
        Z = np.zeros(cs)
        V = gr.Svar(cs, ll, D, Z)
        V.D[0] = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
        V.D[1] = np.array([[7, 8], [9, 10], [11, 12]]).astype(np.float32)
        V.Z = np.array([[13, 14], [15, 16], [17, 18]]).astype(np.float32)
        F = np.array([19, 20, 21, 22]).astype(np.float32)
        result = dyn.vectorize_VF(V, F)
        expected = np.array(
            [
                1,
                3,
                5,
                2,
                4,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
            ]
        ).astype(np.float32)
        np.testing.assert_allclose(result, expected)

    def test_vectorize_VF_torch_2D(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        D = [torch.zeros(cs) for _ in range(2)]
        Z = torch.zeros(cs)
        V = gr.Svar(cs, ll, D, Z)
        V.D[0] = torch.tensor([[1, 2], [3, 4], [5, 6]]).float()
        V.D[1] = torch.tensor([[7, 8], [9, 10], [11, 12]]).float()
        V.Z = torch.tensor([[13, 14], [15, 16], [17, 18]]).float()
        F = torch.tensor([19, 20, 21, 22]).float()
        result = dyn.vectorize_VF(V, F)
        expected = torch.tensor(
            [
                1,
                3,
                5,
                2,
                4,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
            ]
        ).float()
        torch.testing.assert_close(result, expected)

    def test_vectorize_VF_numpy_3D(self):
        cs = (2, 3, 2)
        ll = (1.0, 1.0, 1.0)
        D = [np.zeros(cs) for _ in range(3)]
        Z = np.zeros(cs)
        V = gr.Svar(cs, ll, D, Z)
        V.D[0] = np.array(
            [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
        ).astype(np.float32)
        V.D[1] = np.array(
            [[[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]]]
        ).astype(np.float32)
        V.Z = np.array(
            [[[25, 26], [27, 28], [29, 30]], [[31, 32], [33, 34], [35, 36]]]
        ).astype(np.float32)
        F = np.array([37, 38, 39, 40]).astype(np.float32)
        result = dyn.vectorize_VF(V, F)
        expected = np.array(
            [
                1,
                7,
                3,
                9,
                5,
                11,
                2,
                8,
                4,
                10,
                6,
                12,
                13,
                15,
                17,
                14,
                16,
                18,
                19,
                21,
                23,
                20,
                22,
                24,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
            ]
        ).astype(np.float32)
        np.testing.assert_allclose(result, expected)

    def test_vectorize_VF_torch_3D(self):
        cs = (2, 3, 2)
        ll = (1.0, 1.0, 1.0)
        D = [torch.zeros(cs) for _ in range(3)]
        Z = torch.zeros(cs)
        V = gr.Svar(cs, ll, D, Z)
        V.D[0] = torch.tensor(
            [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
        ).float()
        V.D[1] = torch.tensor(
            [[[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]]]
        ).float()
        V.Z = torch.tensor(
            [[[25, 26], [27, 28], [29, 30]], [[31, 32], [33, 34], [35, 36]]]
        ).float()
        F = torch.tensor([37, 38, 39, 40]).float()
        result = dyn.vectorize_VF(V, F)
        expected = torch.tensor(
            [
                1,
                7,
                3,
                9,
                5,
                11,
                2,
                8,
                4,
                10,
                6,
                12,
                13,
                15,
                17,
                14,
                16,
                18,
                19,
                21,
                23,
                20,
                22,
                24,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
            ]
        ).float()
        torch.testing.assert_close(result, expected)


class TestVectorizeVFMultiF:
    def test_vectorize_VF_multiF_numpy_2D(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        D = [np.zeros(cs) for _ in range(2)]
        Z = np.zeros(cs)
        V = gr.Svar(cs, ll, D, Z)
        V.D[0] = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
        V.D[1] = np.array([[7, 8], [9, 10], [11, 12]]).astype(np.float32)
        V.Z = np.array([[13, 14], [15, 16], [17, 18]]).astype(np.float32)
        F = np.array([19, 20, 21, 22]).astype(np.float32)
        result = dyn.vectorize_VF_multiF(V, [F])
        expected = np.array(
            [
                1,
                3,
                5,
                2,
                4,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
            ]
        ).astype(np.float32)
        np.testing.assert_allclose(result, expected)

    def test_vectorize_VF_multiF_torch_2D(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        D = [torch.zeros(cs) for _ in range(2)]
        Z = torch.zeros(cs)
        V = gr.Svar(cs, ll, D, Z)
        V.D[0] = torch.tensor([[1, 2], [3, 4], [5, 6]]).float()
        V.D[1] = torch.tensor([[7, 8], [9, 10], [11, 12]]).float()
        V.Z = torch.tensor([[13, 14], [15, 16], [17, 18]]).float()
        F = torch.tensor([19, 20, 21, 22]).float()
        result = dyn.vectorize_VF_multiF(V, [F])
        expected = torch.tensor(
            [
                1,
                3,
                5,
                2,
                4,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
            ]
        ).float()
        torch.testing.assert_close(result, expected)

    def test_vectorize_VF_multiF_numpy_3D(self):
        cs = (2, 3, 2)
        ll = (1.0, 1.0, 1.0)
        D = [np.zeros(cs) for _ in range(3)]
        Z = np.zeros(cs)
        V = gr.Svar(cs, ll, D, Z)
        V.D[0] = np.array(
            [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
        ).astype(np.float32)
        V.D[1] = np.array(
            [[[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]]]
        ).astype(np.float32)
        V.Z = np.array(
            [[[25, 26], [27, 28], [29, 30]], [[31, 32], [33, 34], [35, 36]]]
        ).astype(np.float32)
        F = np.array([37, 38]).astype(np.float32)
        result = dyn.vectorize_VF_multiF(V, [F])
        expected = np.array(
            [
                1,
                7,
                3,
                9,
                5,
                11,
                2,
                8,
                4,
                10,
                6,
                12,
                13,
                15,
                17,
                14,
                16,
                18,
                19,
                21,
                23,
                20,
                22,
                24,
            ]
        ).astype(np.float32)
        np.testing.assert_allclose(result, expected)

    def test_vectorize_VF_multiF_torch_3D(self):
        cs = (2, 3, 2)
        ll = (1.0, 1.0, 1.0)
        D = [torch.zeros(cs) for _ in range(3)]
        Z = torch.zeros(cs)
        V = gr.Svar(cs, ll, D, Z)
        V.D[0] = torch.tensor(
            [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
        ).float()
        V.D[1] = torch.tensor(
            [[[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]]]
        ).float()
        V.Z = torch.tensor(
            [[[25, 26], [27, 28], [29, 30]], [[31, 32], [33, 34], [35, 36]]]
        ).float()
        F = torch.tensor([37, 38]).float()
        result = dyn.vectorize_VF_multiF(V, [F])
        expected = torch.tensor(
            [
                1,
                7,
                3,
                9,
                5,
                11,
                2,
                8,
                4,
                10,
                6,
                12,
                13,
                15,
                17,
                14,
                16,
                18,
                19,
                21,
                23,
                20,
                22,
                24,
            ]
        ).float()
        torch.testing.assert_close(result, expected)

    def test_vectorize_VF_multiF_numpy_2D_2F(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        D = [np.zeros(cs) for _ in range(2)]
        Z = np.zeros(cs)
        V = gr.Svar(cs, ll, D, Z)
        V.D[0] = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
        V.D[1] = np.array([[7, 8], [9, 10], [11, 12]]).astype(np.float32)
        V.Z = np.array([[13, 14], [15, 16], [17, 18]]).astype(np.float32)
        F1 = np.array([19, 20, 21, 22]).astype(np.float32)
        F2 = np.array([23, 24, 25, 26]).astype(np.float32)
        result = dyn.vectorize_VF_multiF(V, [F1, F2])
        expected = np.array(
            [
                1,
                3,
                5,
                2,
                4,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
            ]
        ).astype(np.float32)
        np.testing.assert_allclose(result, expected)

    def test_vectorize_VF_multiF_torch_2D_2F(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        D = [torch.zeros(cs) for _ in range(2)]
        Z = torch.zeros(cs)
        V = gr.Svar(cs, ll, D, Z)
        V.D[0] = torch.tensor([[1, 2], [3, 4], [5, 6]]).float()
        V.D[1] = torch.tensor([[7, 8], [9, 10], [11, 12]]).float()
        V.Z = torch.tensor([[13, 14], [15, 16], [17, 18]]).float()
        F1 = torch.tensor([19, 20, 21, 22]).float()
        F2 = torch.tensor([23, 24, 25, 26]).float()
        result = dyn.vectorize_VF_multiF(V, [F1, F2])
        expected = torch.tensor(
            [
                1,
                3,
                5,
                2,
                4,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
            ]
        ).float()
        torch.testing.assert_close(result, expected)


class TestUnvectorizeVF:
    def test_unvectorize_VF_numpy_2D(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        D = [np.zeros(cs) for _ in range(2)]
        Z = np.zeros(cs)
        V = gr.Svar(cs, ll, D, Z)
        V.D[0] = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
        V.D[1] = np.array([[7, 8], [9, 10], [11, 12]]).astype(np.float32)
        V.Z = np.array([[13, 14], [15, 16], [17, 18]]).astype(np.float32)
        F = np.array([19, 20, 21, 22]).astype(np.float32)

        vec = np.array(
            [
                1,
                3,
                5,
                2,
                4,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
            ]
        ).astype(np.float32)
        V_res, F_res = dyn.unvectorize_VF(vec, 4, cs, ll)
        np.testing.assert_allclose(V_res.D[0], V.D[0])
        np.testing.assert_allclose(V_res.D[1], V.D[1])
        np.testing.assert_allclose(V_res.Z, V.Z)
        np.testing.assert_allclose(F_res, F)

    def test_unvectorize_VF_torch_2D(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        D = [torch.zeros(cs) for _ in range(2)]
        Z = torch.zeros(cs)
        V = gr.Svar(cs, ll, D, Z)
        V.D[0] = torch.tensor([[1, 2], [3, 4], [5, 6]]).float()
        V.D[1] = torch.tensor([[7, 8], [9, 10], [11, 12]]).float()
        V.Z = torch.tensor([[13, 14], [15, 16], [17, 18]]).float()
        F = torch.tensor([19, 20, 21, 22]).float()
        vec = torch.tensor(
            [
                1,
                3,
                5,
                2,
                4,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
            ]
        ).float()
        V_res, F_res = dyn.unvectorize_VF(vec, 4, cs, ll)
        torch.testing.assert_close(V_res.D[0], V.D[0])
        torch.testing.assert_close(V_res.D[1], V.D[1])
        torch.testing.assert_close(V_res.Z, V.Z)
        torch.testing.assert_close(F_res, F)

    def test_unvectorize_VF_numpy_3D(self):
        cs = (2, 3, 2)
        ll = (1.0, 1.0, 1.0)
        D = [np.zeros(cs) for _ in range(3)]
        Z = np.zeros(cs)
        V = gr.Svar(cs, ll, D, Z)
        V.D[0] = np.array(
            [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
        ).astype(np.float32)
        V.D[1] = np.array(
            [[[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]]]
        ).astype(np.float32)
        V.Z = np.array(
            [[[25, 26], [27, 28], [29, 30]], [[31, 32], [33, 34], [35, 36]]]
        ).astype(np.float32)
        F = np.array([37, 38, 39, 40]).astype(np.float32)
        vec = np.array(
            [
                1,
                7,
                3,
                9,
                5,
                11,
                2,
                8,
                4,
                10,
                6,
                12,
                13,
                15,
                17,
                14,
                16,
                18,
                19,
                21,
                23,
                20,
                22,
                24,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
            ]
        ).astype(np.float32)
        V_res, F_res = dyn.unvectorize_VF(vec, 4, cs, ll)
        np.testing.assert_allclose(V_res.D[0], V.D[0])
        np.testing.assert_allclose(V_res.D[1], V.D[1])
        np.testing.assert_allclose(V_res.D[2], V.D[2])
        np.testing.assert_allclose(V_res.Z, V.Z)
        np.testing.assert_allclose(F_res, F)

    def test_unvectorize_VF_torch_3D(self):
        cs = (2, 3, 2)
        ll = (1.0, 1.0, 1.0)
        D = [torch.zeros(cs) for _ in range(3)]
        Z = torch.zeros(cs)
        V = gr.Svar(cs, ll, D, Z)
        V.D[0] = torch.tensor(
            [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
        ).float()
        V.D[1] = torch.tensor(
            [[[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]]]
        ).float()
        V.Z = torch.tensor(
            [[[25, 26], [27, 28], [29, 30]], [[31, 32], [33, 34], [35, 36]]]
        ).float()
        F = torch.tensor([37, 38, 39, 40]).float()
        vec = torch.tensor(
            [
                1,
                7,
                3,
                9,
                5,
                11,
                2,
                8,
                4,
                10,
                6,
                12,
                13,
                15,
                17,
                14,
                16,
                18,
                19,
                21,
                23,
                20,
                22,
                24,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
            ]
        ).float()
        V_res, F_res = dyn.unvectorize_VF(vec, 4, cs, ll)
        torch.testing.assert_close(V_res.D[0], V.D[0])
        torch.testing.assert_close(V_res.D[1], V.D[1])
        torch.testing.assert_close(V_res.D[2], V.D[2])
        torch.testing.assert_close(V_res.Z, V.Z)
        torch.testing.assert_close(F_res, F)


class TestUnvectorizeVFMultiF:
    def test_unvectorize_VF_multiF_numpy_2D(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        D = [np.zeros(cs) for _ in range(2)]
        Z = np.zeros(cs)
        V = gr.Svar(cs, ll, D, Z)
        V.D[0] = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
        V.D[1] = np.array([[7, 8], [9, 10], [11, 12]]).astype(np.float32)
        V.Z = np.array([[13, 14], [15, 16], [17, 18]]).astype(np.float32)
        F1 = np.array([19, 20, 21, 22]).astype(np.float32)
        F2 = np.array([23, 24, 25, 26]).astype(np.float32)
        vec = dyn.vectorize_VF_multiF(V, [F1, F2])
        V_res, F_res = dyn.unvectorize_VF_multiF(vec, [(4,), (4,)], cs, ll)
        np.testing.assert_allclose(V_res.D[0], V.D[0])
        np.testing.assert_allclose(V_res.D[1], V.D[1])
        np.testing.assert_allclose(V_res.Z, V.Z)
        np.testing.assert_allclose(F_res, [F1, F2])

    def test_unvectorize_VF_multiF_torch_2D(self):
        cs = (3, 2)
        ll = (1.0, 1.0)
        D = [torch.zeros(cs) for _ in range(2)]
        Z = torch.zeros(cs)
        V = gr.Svar(cs, ll, D, Z)
        V.D[0] = torch.tensor([[1, 2], [3, 4], [5, 6]]).float()
        V.D[1] = torch.tensor([[7, 8], [9, 10], [11, 12]]).float()
        V.Z = torch.tensor([[13, 14], [15, 16], [17, 18]]).float()
        F1 = torch.tensor([19, 20, 21, 22]).float()
        F2 = torch.tensor([23, 24, 25, 26]).float()
        vec = dyn.vectorize_VF_multiF(V, [F1, F2])
        V_res, F_res = dyn.unvectorize_VF_multiF(vec, [(4,), (4,)], cs, ll)
        torch.testing.assert_close(V_res.D[0], V.D[0])
        torch.testing.assert_close(V_res.D[1], V.D[1])
        torch.testing.assert_close(V_res.Z, V.Z)
        torch.testing.assert_close(F_res, [F1, F2])


class TestBuildHOperatorMatrix:
    def test_build_H_operator_matrix_numpy_2D(self):
        H = np.array([[1, 2], [3, 4]])
        dx = 1
        res = dyn.build_H_operator_matrix(H, dx).todense()
        np.testing.assert_allclose(
            res,
            np.array(
                [
                    [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        )

    def test_build_H_operator_matrix_torch_2D(self):
        H = torch.tensor([[1, 2], [3, 4]])
        dx = 1
        res = dyn.build_H_operator_matrix(torch.tensor(H).float(), dx)
        torch.testing.assert_close(
            res,
            torch.tensor(
                [
                    [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ).float(),
            check_dtype=False,
        )

    def test_build_H_operator_matrix_numpy_3D(self):
        H = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        dx = 1
        res = dyn.build_H_operator_matrix(H, dx).todense()
        np.testing.assert_allclose(
            res,
            np.array(
                [
                    [
                        1,
                        0,
                        3,
                        0,
                        2,
                        0,
                        4,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        5,
                        0,
                        7,
                        0,
                        6,
                        0,
                        8,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                ]
            ),
        )

    def test_build_H_operator_matrix_torch_3D(self):
        H = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        dx = 1
        res = dyn.build_H_operator_matrix(H, dx)
        torch.testing.assert_close(
            res,
            torch.tensor(
                [
                    [
                        1,
                        0,
                        3,
                        0,
                        2,
                        0,
                        4,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        5,
                        0,
                        7,
                        0,
                        6,
                        0,
                        8,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                ]
            ),
            check_dtype=False,
        )


class TestBuildHOperatorMatrixMulti:
    def test_build_H_operator_matrix_multi_numpy_2D(self):
        H1 = np.array([[1, 2], [3, 4]])
        H2 = np.array([[5, 6], [7, 8]])
        dx = 1
        res = dyn.build_H_operator_matrix_multi([H1, H2], dx).todense()
        np.testing.assert_allclose(
            res,
            np.array(
                [
                    [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
                    [5, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 7, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        )


class TestBuildHandFlattenArray:
    def test_build_H_and_flatten_array_numpy_small_2D(self):
        H = np.array([[1, 2], [3, 4]])
        dx = 1
        H_mat = dyn.build_H_operator_matrix(H, dx)
        a = np.array([[5, 6], [7, 8]]).astype(np.float32)
        fastest_axis = 0
        b = dyn.flatten_array(a, fastest_axis)
        b = np.concatenate([b] + [np.zeros(4)] * 2)
        c = H_mat @ b
        np.testing.assert_allclose((H * a * dx).sum(axis=1), c)

    def test_build_H_and_flatten_array_torch_small_2D(self):
        H = torch.tensor([[1, 2], [3, 4]]).float()
        dx = 1
        H_mat = dyn.build_H_operator_matrix(H, dx)
        a = torch.tensor([[5, 6], [7, 8]]).float()
        fastest_axis = 0
        b = dyn.flatten_array(a, fastest_axis)
        b = torch.cat([b, torch.zeros(4), torch.zeros(4)], dim=0)
        c = H_mat @ b
        torch.testing.assert_close((H * a * dx).sum(axis=1), c)

    def test_build_H_and_flatten_array_numpy_small_3D(self):
        H = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        dx = 1
        H_mat = dyn.build_H_operator_matrix(H, dx)
        a = np.array([[[5, 6], [7, 8]], [[9, 10], [11, 12]]]).astype(np.float32)
        fastest_axis = 0
        b = dyn.flatten_array(a, fastest_axis)
        b = np.concatenate([b] + [np.zeros(8)] * 3)
        c = H_mat @ b
        np.testing.assert_allclose((H * a * dx).sum(axis=(1, 2)), c)

    def test_build_H_and_flatten_array_torch_small_3D(self):
        H = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).float()
        dx = 1
        H_mat = dyn.build_H_operator_matrix(H, dx)
        a = torch.tensor([[[5, 6], [7, 8]], [[9, 10], [11, 12]]]).float()
        fastest_axis = 0
        b = dyn.flatten_array(a, fastest_axis)
        b = torch.cat([b] + [torch.zeros(8)] * 3, dim=0)
        c = H_mat @ b
        torch.testing.assert_close((H * a * dx).sum(axis=(1, 2)), c)

    def test_build_H_and_flatten_array_numpy_large_random(self):
        H = np.random.rand(3, 4, 5, 6).astype(np.float32)
        dx = 0.5
        H_mat = dyn.build_H_operator_matrix(H, dx)
        a = np.random.rand(3, 4, 5, 6).astype(np.float32)
        fastest_axis = 0
        b = dyn.flatten_array(a, fastest_axis)
        b = np.concatenate([b] + [np.zeros(360)] * 4)
        c = H_mat @ b
        np.testing.assert_allclose((H * a * dx).sum(axis=(1, 2, 3)), c)

    def test_build_H_and_flatten_array_torch_large_random(self):
        H = torch.rand(3, 4, 5, 6).float()
        dx = 0.5
        H_mat = dyn.build_H_operator_matrix(H, dx)
        a = torch.rand(3, 4, 5, 6).float()
        fastest_axis = 0
        b = dyn.flatten_array(a, fastest_axis)
        b = torch.cat([b] + [torch.zeros(360)] * 4, dim=0)
        c = H_mat @ b
        torch.testing.assert_close((H * a * dx).sum(axis=(1, 2, 3)), c)


class TestIofN:
    def test_I_of_N_shape(self):
        """Test that I_of_N(N) has the correct shape (N x (N+1))."""
        for N in [1, 2, 3, 5]:
            M = dyn.I_of_N(N)
            assert M.shape == (
                N,
                N + 1,
            ), f"Expected shape {(N, N+1)} for N={N}, but got {M.shape}"

    def test_I_of_N_entries(self):
        """
        Test that each row k contains exactly two 0.5 entries,
        at positions [k, k+1], and everything else is zero.
        """
        for N in [1, 2, 3]:
            M = dyn.I_of_N(N)
            for k in range(N):
                # Create an array of zeros of length N+1, set positions k, k+1 to 0.5
                expected_row = np.zeros(N + 1, dtype=float)
                expected_row[k] = 0.5
                expected_row[k + 1] = 0.5
                np.testing.assert_array_almost_equal(
                    M[k],
                    expected_row,
                    err_msg=f"Row {k} of I_of_N({N}) not as expected.",
                )

    def test_I_of_N_numerical_values(self):
        """
        Simple check on sum of all entries.
        For N, the sum of all entries in I_of_N(N) should be N (since each row has two 0.5s).
        """
        for N in [1, 2, 5]:
            M = dyn.I_of_N(N)
            total_sum = np.sum(M)
            expected_sum = N  # N rows * 0.5 + 0.5 = N
            assert (
                abs(total_sum - expected_sum) < 1e-12
            ), f"Sum of entries in I_of_N({N}) should be {expected_sum}, got {total_sum}"


class TestBuildBigBlock:
    def test_build_big_block_single_dimension(self):
        """
        If N_list has a single element, say [2], then we only have Q[0] and 2 * the 2x2 identity,
        repeated prod/N_0 = 2//2 = 1 time, so it's just two 2x2 blocks.
        """
        N_list = [2]
        nx = be.NumpyBackend_ext()
        result = dyn.build_big_block(N_list, nx).todense()
        # result should be 2x2
        assert result.shape == (
            4,
            4,
        ), f"Expected a single 4x4 block, got {result.shape}"

        # Let's compute what we expect: Q[0] = I_of_N(2)@I_of_N(2).T + I(2)
        I2 = dyn.I_of_N(2)  # shape (2,3)
        # I2@I2.T -> shape (2,2)
        expected_Q0 = I2 @ I2.T + np.eye(2)
        expected = sp.linalg.block_diag(expected_Q0, 2 * np.eye(2))  # shape 4x4
        # assert False, f"{expected}"
        # Compare with the result (it should match exactly, as only one block).
        np.testing.assert_array_almost_equal(
            result,
            expected,
            err_msg="Single block for build_big_block([2]) does not match expected Q.",
        )

    def test_build_big_block_two_dimensions(self):
        """
        N_list = [1, 2]
        - product of [1,2] is 2.
        - Q[0] with dimension 1x1 is repeated 2/1 = 2 times (block size 2 in the diagonal).
        - Q[1] with dimension 2x2 is repeated 2/2 = 1 time (block size 2 in the diagonal).
        - Finally, we have a N_list[-1] x N_list[-1] = 2x2 identity x 2 in the diagonal.
        So final result is a 4-block diagonal with shape (1*2 + 2*1 + 2) = 6 x 6.
        """
        N_list = [1, 2]
        nx = be.NumpyBackend_ext()

        result = dyn.build_big_block(N_list, nx).todense()

        # Expected shape is 4x4 (since Q[0] repeated 2 times -> 1+1=2 rows, Q[1] repeated once -> 2 rows, total=4)
        # assert result.shape == (4, 4), f"Expected a 4x4 block, got {result.shape}"

        # We know Q[0] is 1x1 => I_of_N(1) = [0.5, 0.5], so Q[0] = (I_of_N(1)@I_of_N(1).T) + I(1).
        I1 = dyn.I_of_N(1)  # shape (1,2)
        Q0 = I1 @ I1.T + np.eye(1)  # 1x1
        # Q[1] is 2x2
        I2 = dyn.I_of_N(2)
        Q1 = I2 @ I2.T + np.eye(2)

        # The block structure of 'result' should be diag(Q0, Q0, Q1).
        # Let's build that structure manually and compare.
        # We can do this with block_diag in Python:

        expected = sp.linalg.block_diag(
            Q0, Q0, Q1, 2 * np.eye(2)
        )  # shape 1+1+2+2 = 6x6

        np.testing.assert_array_almost_equal(
            result,
            expected,
            err_msg="Block structure for build_big_block([1,2]) does not match expected diag(Q0,Q0,Q1).",
        )

    def test_build_big_block_larger_example(self):
        """
        A slightly larger example, just to test shape and basic numeric consistency.
        We'll do N_list = [2, 3].
        product = 6.
        - Q[0] is 2x2 repeated 6/2=3 times -> 2+2+2 = 6 total rows from Q[0] blocks
        - Q[1] is 3x3 repeated 6/3=2 times -> 3+3 = 6 total rows from Q[1] blocks
        - Finally, we have a 3x3 identity x 2 in the diagonal -> 6 rows.
        Overall shape = 18x18.
        """
        N_list = [2, 3]
        nx = be.NumpyBackend_ext()
        result = dyn.build_big_block(N_list, nx).todense()
        assert result.shape == (18, 18), f"Expected a 12x12 block, got {result.shape}"

        # We won't check all numeric entries here; we just trust the previous tests
        # for correctness. But let's do a few quick checks on sums or diagonal positivity.
        diag_elems = np.diag(result)
        assert np.all(
            diag_elems > 0
        ), "All diagonal elements should be positive (due to identity + something)."

    @pytest.mark.parametrize("N_list", [([1]), ([1, 1]), ([2, 2]), ([1, 2, 3])])
    def test_build_big_block_general(self, N_list):
        """
        Parametric test: just ensure the final block matrix is square of
        dimension sum_j ( (prod_i N_i / N_j) * N_j ) = ((len(N_list) + 1) * prod_i N_i).
        """
        from math import prod

        nx = be.NumpyBackend_ext()

        p = prod(N_list)
        expected_dim = (len(N_list) + 1) * p
        result = dyn.build_big_block(N_list, nx)
        assert result.shape == (
            expected_dim,
            expected_dim,
        ), f"For N_list={N_list}, expected block shape {expected_dim}x{expected_dim}, got {result.shape}"


class TestStepDR:
    def test_step_DR_numpy_small(self):
        def prox1(y, x):
            y.U.D[0][...] = 2 * x.U.D[0]

        def prox2(z, w):
            pass

        rho0 = np.array([1, 2]).astype(np.float32)
        rho1 = np.array([5, 6]).astype(np.float32)
        T = 2
        ll = (1.0, 1.0)
        w, x, y, z = [gr.CSvar(rho0, rho1, T, ll) for _ in range(4)]
        w.U.D[0] = np.array([[1, 1], [1, 1], [1, 1]]).astype(np.float32)
        z.U.D[0] = np.array([[0, 0], [0, 0], [0, 0]]).astype(np.float32)

        assert isinstance(x.U, gr.Svar)

        w, x, y, z = dyn.stepDR(w, x, y, z, prox1, prox2, 1.0)
        np.testing.assert_allclose(
            y.U.D[0], np.array([[-2.0, -2.0], [-2.0, -2.0], [-2.0, -2.0]])
        )


class TestPreComuputePPT:
    def test_precomputePPT_numpy_small(self):
        H = np.array([[1, 2], [3, 4]]).astype(np.float32)
        cs = (2, 2)
        ll = (1.0, 1.0)
        PPT = dyn.precomputePPT(H, list(cs), ll)
        expected = np.array(
            [
                [1.5, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, 0],
                [0.25, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.5],
                [0, 0, 1.5, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
                [0, 0, 0.25, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2],
                [0, 0, 0, 0, 1.5, 0.25, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0.25, 1.5, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1.5, 0.25, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0.25, 1.5, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                [-0.5, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.25, 0],
                [0, -1.5, 0, -2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6.25],
            ]
        )
        np.testing.assert_allclose(PPT.todense(), expected)


class TestPreComputePPTPeriodic:
    def test_precomputePPT_numpy_small_periodic(self):
        H = np.array([[1, 2], [3, 4]]).astype(np.float32)
        cs = (2, 2)
        ll = (1.0, 1.0)
        PPT = dyn.precomputePPT(H, list(cs), ll, periodic=True)
        expected = np.array(
            [
                [1.5, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, 0],
                [0.25, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.5],
                [0, 0, 1.5, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
                [0, 0, 0.25, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2],
                [0, 0, 0, 0, 1.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0.5, 1.5, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1.5, 0.5, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0.5, 1.5, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                [-0.5, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.25, 0],
                [0, -1.5, 0, -2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6.25],
            ]
        )
        np.testing.assert_allclose(PPT.todense(), expected)

    def test_precomputePPT_numpy_small_periodic_multi(self):
        H1 = np.array([[1, 2], [3, 4]]).astype(np.float32)
        H2 = np.array([[5, 6], [7, 8]]).astype(np.float32)
        cs = (2, 2)
        ll = (1.0, 1.0)
        PPT = dyn.precomputePPT([H1, H2], list(cs), ll, periodic=True)
        expected = np.array(
            [
                [1.5, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, 0, -2.5, 0],
                [0.25, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.5, 0, -3.5],
                [0, 0, 1.5, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -3, 0],
                [0, 0, 0.25, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -4],
                [0, 0, 0, 0, 1.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0.5, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0.5, 1.5, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
                [-0.5, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.25, 0, 4.25, 0],
                [0, -1.5, 0, -2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6.25, 0, 13.25],
                [-2.5, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4.25, 0, 15.25, 0],
                [0, -3.5, 0, -4.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13.25, 0, 28.25],
            ]
        )
        np.testing.assert_allclose(PPT.todense(), expected)


class TestComputeGeodesic:

    def test_computeGeodesic_numpy_samedist(self):
        # rho0 = rho1, so everything should be one and the momentum should be zero
        rho0 = np.array([1, 1, 1]).astype(np.float32)
        rho1 = np.array([1, 1, 1]).astype(np.float32)
        T = 5
        ll = (1.0, 1.0)
        z, list = dyn.computeGeodesic(rho0, rho1, T, ll)
        np.testing.assert_allclose(z.U.D[0], np.ones((T + 1, 3)))
        np.testing.assert_allclose(z.U.D[1], np.zeros((T, 4)))
        np.testing.assert_allclose(z.U.Z, np.zeros((T, 3)))

    def test_computeGeodesic_numpy_samedist_constraint(self):
        # rho0 = rho1, so everything should be one and the momentum should be zero
        rho0 = np.array([1, 1, 1]).astype(np.float32)
        rho1 = np.array([1, 1, 1]).astype(np.float32)
        T = 5
        ll = (1.0, 1.0)
        H = np.ones((T, 3), dtype=np.float32)
        F = np.array([1, 1, 1, 1, 1], dtype=np.float32)
        z, list = dyn.computeGeodesic(rho0, rho1, T, ll, H, F)
        np.testing.assert_allclose(z.U.D[0], np.ones((T + 1, 3)))
        np.testing.assert_allclose(z.U.D[1], np.zeros((T, 4)))
        np.testing.assert_allclose(z.U.Z, np.zeros((T, 3)))
