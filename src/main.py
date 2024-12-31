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
from jax.numpy.linalg import norm, inv
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
    P_yy = 9.0 * jnp.identity(3)
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
    g_dot_B = lambda t, g: jnp.concat([
        x_dot(g[0:6]), 
        jnp.reshape(phi_dot(g[0:6], jnp.reshape(g[6:42], [6, 6])), [36]),
        jnp.reshape(S_dot(g[0:6], jnp.reshape(g[42:48], [6, 1])), [6]),
        ])  

    # Initial values
    x_bar_0_B = jnp.array([data.rx[0], data.ry[0], data.rz[0], data.vx[0], data.vy[0], data.vz[0]])
    phi_00_B = jnp.identity(6)
    S_00_B = jnp.zeros([6,1])
    g_0_B = jnp.concat([x_bar_0_B, jnp.reshape(phi_00_B, [36]), jnp.reshape(S_00_B, [6])])

    # solution_B = solve_ivp(g_dot_B, (data.t[0], data.t[1]), g_0_B, method="RK45", t_eval=[data.t[0], data.t[1]], atol=1e-6)
    # print(solution_B)
    # print(array_to_latex_matrix(jnp.round(jnp.reshape(solution_B.y[6:42, 1],[6,6]), 10)))
    # print(array_to_latex_matrix(jnp.round(jnp.reshape(solution_B.y[42:48, 1],[6,1]), 10)))
    # fmt: on

    # Part C
    # Functions
    h = lambda x, u: norm(x[0:3] - u, axis=0).reshape(-1, 1)
    # Derivatives
    g_dot_g = (
        lambda t, g: jnp.concat(  # No sensitivity matrix
            [
                x_dot(g[0:6]),
                jnp.reshape(phi_dot(g[0:6], jnp.reshape(g[6:42], [6, 6])), [36]),
            ]
        )
    )
    H_xn = lambda x, u: jnp.squeeze(jacobian(h)(x, u))

    # Initial conditions
    x_bar = list([jnp.array([data.rx[0], data.ry[0], data.rz[0], data.vx[0], data.vy[0], data.vz[0]]).reshape(-1,1)])
    
    z_bar = jnp.array(data.CA_range).reshape(100,-1,1)
    z_bar = [z_bar[i][~jnp.all(z_bar[i] == 0, axis=1)] for i in range(z_bar.shape[0])]
    # print(z_bar[0][0])
       
    r_gps = jnp.array([jnp.column_stack([data.rx_gps, data.ry_gps, data.rz_gps])]).reshape(100,3,-1)
    r_gps = [r_gps[i][:, ~np.all(r_gps[i] == 0, axis=0)] for i in range(r_gps.shape[0])]
    # print(r_gps[0][:,0])

    P = list([jnp.array(P_xx)])
    R = list([0.009*jnp.identity(len(z_bar[0]))])
    
    dz_bar = list([z_bar[0]-h(x_bar[0], r_gps[0])])
    H = list([H_xn(x_bar[0], r_gps[0])])
    K = list([P[0] @ H[0].T @ inv(H[0] @ P[0] @ H[0].T + R[0])])
    x_hat = list([x_bar[0] + K[0] @ dz_bar[0]])
    P_hat = list([(jnp.identity(6) - K[0] @ H[0]) @ P[0]])
    
    # Propagation loop
    for i in range(1):
        print(i)
    
    return
    # print(h(x_bar[0], len(z_bar)))

    # Propagation loop
    # for i in range(len(data.t) - 1):
    #     phi_00 = jnp.identity(6)
    #     g_0 = jnp.concat([x_hat, jnp.reshape(phi_00, [36])])
    #     solution = solve_ivp(
    #     g_dot,
    #     (data.t[i], data.t[i + 1]),
    #     g_0,
    #     method="RK45",
    #     t_eval=[data.t[i + 1]],
    #     atol=1e-6,
    # )


if __name__ == "__main__":
    main()
