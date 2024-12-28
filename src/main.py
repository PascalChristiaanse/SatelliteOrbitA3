"""_summary_"""

# Formatter settings
# ruff: noqa: E731, F841

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

    # print("P_yy", P_yy)
    # print("P_xx", P_xx)

    # Part B

    # fmt: off
    # Constants (should really be in a vector P but not worth the effort as we're only using mu) note to resolve: solve_ivp can use the "args" parameter to pass p to the functions!
    mu = 3.986e5
    # Functions
    U = lambda r: mu / jnp.linalg.norm(r)
    f = lambda x: jnp.array([x[3], x[4], x[5], *grad(U)(x[0:3])])  # also x_dot

    # Derivatives    
    x_dot = f
    F_x = lambda x: jacobian(f)(x) # jacobian of f wrt x
    F_p = lambda x: jnp.array([0,0,0,-x[0]/norm(x[0:3]),-x[1]/norm(x[0:3]),-x[2]/norm(x[0:3])]).reshape(-1,1) # noqa: E731
    phi_dot = lambda x, phi: F_x(x) @ phi  
    S_dot = lambda x, S: F_x(x) @ S + F_p(x) # really part of C but okay

    # Initial values
    x_0 = jnp.array([data.rx[0], data.ry[0], data.rz[0], data.vx[0], data.vy[0], data.vz[0]])
    phi_00 = jnp.identity(6)
    S_00 = jnp.zeros([6,1])

    g_0 = jnp.concat([x_0, jnp.reshape(phi_00, [36]), jnp.reshape(S_00, [6])])
    g_dot = lambda t, g: jnp.concat([
        x_dot(g[0:6]), 
        jnp.reshape(phi_dot(g[0:6], jnp.reshape(g[6:42], [6, 6])), [36]),
        jnp.reshape(S_dot(g[0:6], jnp.reshape(g[42:48], [6, 1])), [6]),
        ])  
    # fmt: on

    solution = solve_ivp(
        g_dot,
        (data.t[0], data.t[1]),
        g_0,
        method="RK45",
        t_eval=[data.t[0], data.t[1]],
        atol=1e-6,
    )
    print(solution)
    print(array_to_latex_matrix(jnp.round(jnp.reshape(solution.y[6:, 1],[6,6]), 10)))


if __name__ == "__main__":
    main()
