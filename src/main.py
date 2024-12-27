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


if __name__ == "__main__":
    main()
