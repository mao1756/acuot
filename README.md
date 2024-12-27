# Results of the experiments
Here, we record the results of our experiments so far. For the information about the repository, see below.

## Overall results for the proximal algorithm
| Constraint        | Status    |            Note                 |   Folder            |
|-------------------|-----------|-------------------------------- |--------------------------------|
| Unconstrained     | Success   | Recreation of Chizat et al. (2018) and more |[link](exploratory/proximal/unconstrained) | 
| Total Mass        | Success   | SHK distance is covered here    | [link](exploratory/proximal/total_mass) |
| Barrier           | Partial   | Moving barrier is more unstable | [link](exploratory/proximal/barrier) |
| Convex Sets       | N/A       | Not implemented yet             | [link](exploratory/proximal/convex_sets) |

### Unconstrained experiments

| Experiment        | Status    |            Note                 |   Folder            |
|-------------------|-----------|-------------------------------- |--------------------------------|
| Transport of Diracs   | Fail   |   | [link](exploratory/proximal/unconstrained/[FAILS]transport_of_diracs.ipynb) |
| 1D Gaussian Bump   | Success   | Recreation of Chizat et al. (2018)  | [link](exploratory/proximal/unconstrained/[WORKS]chizat2018_1d_gaussian_bump.ipynb) |
| Deformation of A Ring   | Success   | Recreation of Chizat et al. (2018)  | [link](exploratory/proximal/unconstrained/[WORKS]chizat2018_deformation_of_ring.ipynb) |
| CIRM Special   | Success   | Presented at CIRM for Dr. Michor's birthday  | [link](exploratory/proximal/unconstrained/[WORKS]cirm_special.ipynb) |
| Poisson Solver Test | Success | The Poisson solver is $O(1/N^2)$ as expected | [link](exploratory/proximal/unconstrained/[FAILS]test_poisson_error.ipynb) |

### Total mass constraint

| Experiment        | Status    |            Note                 |   Folder            |
|-------------------|-----------|-------------------------------- |--------------------------------|
| SHK vs Constant Mass WFR    | Success   |  Comparing theoretical SHK geodesic & distance to our algorithm  | [link](exploratory/proximal/total_mass/[WORKS]constant_mass_vs_SHK.ipynb) |
| Shrink and enlarge the same distribution    | Success   |   | [link](exploratory/proximal/total_mass/[WORKS]shrink_enlarge_same_dist.ipynb) |
| Ours vs Jing et al. | Partial | Larger domain induces instability | [link](exploratory/proximal/total_mass/[PARTIAL]SHK_ours_vs_jing_gaussian.ipynb) |

### Barrier Constraint

| Experiment        | Status    |            Note                 |   Folder            |
|-------------------|-----------|-------------------------------- |--------------------------------|
| One Stationary Wall    | Success   |    | [link](exploratory/proximal/barrier/[WORKS]going_around_one_stationary_wall.ipynb) |
| Two Stationary Walls    | Success   |    | [link](exploratory/proximal/barrier/[WORKS]going_around_two_stationary_wall.ipynb) |
| One Stationary One Moving | Partial | Too big step size for the constraining function $H(t,x)$ leads to divergence| [link](exploratory/proximal/barrier/[PARTIAL]one_stationary_one_moving.ipynb)
| Vanishing Walls | Partial | Same as above and the violation of continuity equation is higher than others | [link](exploratory/proximal/barrier/[PARTIAL]gradually_vanishing_walls.ipynb)

### Convex Curve Constraint

| Experiment        | Status    |            Note                 |   Folder            |



# Python Dynamic Optimal Transport

The aim of this project is to implement various methods for the dynamic formulation of optimal transport, namely the optimization of the kinetic energy on the space of all solutions on the continuity equation. The most basic dynamic OT problem is the Benamou-Brenier formulation of the Wasserstein distance. Given probability densities $\rho_0(x)$ and $\rho_1(x)$ on $\mathbb{R}^d$, the square of the Wasserstein 2-distance is equal to the solution of the following optimization problem:

$$\begin{align} &\textrm{minimize } \int_{0}^{1}\int_{\mathbb{R}^d}\rho(t,x)\|v(t,x)\|^2dxdt \\\ &\textrm{subject to } \partial_t \rho + \textrm{div}(\rho v ) = 0, \rho(0,x)=\rho_0(x),\rho(1,x)=\rho_1(x)\end{align} \tag*{}$$

We will implement the algorithm to solve this problem and its generalizations.

# Features

Our main aim is to implement the dynamical formulation of the Wasserstein-Fisher-Rao optimal transport, where we solve the following problem:

$$\begin{align} &\textrm{minimize } \frac{1}{2}\left(\int_{0}^{1}\int_{\mathbb{R}^d}\rho(t,x)\|v(t,x)\|^2dxdt+\delta^2\int_{0}^{1}\int_{\mathbb{R}^d}\rho(t,x)\|z(t,x)\|^2dxdt\right) \\\ &\textrm{subject to } \partial_t \rho + \textrm{div}(\rho v ) = \rho z, \rho(0,x)=\rho_0(x),\rho(1,x)=\rho_1(x)\end{align} \tag*{}$$

We approach this problem by the `proximal` algorithm originally by Chizat et al. (2018).