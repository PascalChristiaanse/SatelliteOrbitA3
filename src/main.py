"""_summary_"""

# Add root folder
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Libs
import numpy as np
import jax.numpy as jnp
import jax
from jax import grad, jacobian
from jax.numpy.linalg import norm
from jax.numpy import pow

from scipy.integrate import solve_ivp

# Data
from data.read_txt_python import Data

# Solvers


# Plotters
from OrbitPlotter import OrbitPlotter


def array_to_latex_matrix(arr):
    # Start the LaTeX matrix environment
    latex_str = "\\begin{bmatrix}\n"

    # Iterate over the rows of the array
    for row in arr:
        # Join the elements of the row with ' & ' and append to the LaTeX string
        latex_str += " & ".join(map(str, row)) + " \\\\\n"

    # End the matrix environment
    latex_str += "\\end{bmatrix}"

    return latex_str


def main():
    data = Data()
    orbit_pltr = OrbitPlotter()
    # Obtain reference solution
    ref_sol = []
    for i in range(len(data.rx)):
        ref_sol.append(
            [
                data.rx[i],
                data.ry[i],
                data.rz[i],
                data.vx[i],
                data.vy[i],
                data.vz[i],
            ]
        )
    ref_sol = np.array(ref_sol)

    # Plot reference solution

    orbit_pltr.add_orbit(ref_sol[:, :-3], name="reference", color="blue")
    # orbit_pltr.plot()

    # Part A
    # Define covariance matrices
    P_yy = np.array([[9, 0, 0], [0, 9, 0], [0, 0, 9]])
    P_xx = np.array(
        [
            [4, 0, 0, 0.14, 0, 0],
            [0, 4, 0, 0, 0.14, 0],
            [0, 0, 4, 0, 0, 0.14],
            [0.14, 0, 0, 0.01, 0, 0],
            [0, 0.14, 0, 0, 0.01, 0],
            [0, 0, 0.14, 0, 0, 0.01],
        ]
    )

    print("P_yy", P_yy)
    print("P_xx", P_xx)

    # Part B
    # Derivatives
    # fmt: off
    mu = 3.986e5
    U = lambda r: mu / jnp.linalg.norm(r)  # noqa: E731
    f = lambda x: jnp.array([x[3], x[4], x[5], *grad(U)(x[0:3])])  # noqa: E731 also x_dot
    F = lambda x: jacobian(f)(x)  # noqa: E731
    x_dot = f  # noqa: E731
    phi_dot = lambda x, phi: F(x) @ phi  # noqa: E731

    # Initial values
    x_0 = jnp.array([data.rx[0], data.ry[0], data.rz[0], data.vx[0], data.vy[0], data.vz[0]])
    phi_00 = jnp.identity(6)

    g_0 = jnp.concat([x_0, jnp.reshape(phi_00, [36])])
    g_dot = lambda t, g: jnp.concat([x_dot(g[0:6]), jnp.reshape(phi_dot(g[0:6], jnp.reshape(g[6:], [6, 6])), [36])])  # noqa: E731
    # fmt: on

    solution = solve_ivp(g_dot, (data.t[0], data.t[1]), g_0, method="RK45", t_eval=[data.t[0], data.t[1]], atol=1e-6)
    print(solution)
    print(array_to_latex_matrix(jnp.round(jnp.reshape(solution.y[6:, 1],[6,6]), 10)))


if __name__ == "__main__":
    main()
