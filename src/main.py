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
from jax import grad, jacobian, jit
from jax.numpy.linalg import norm, inv
from jax.numpy import pow

from scipy.integrate import solve_ivp

# Data
from data.read_txt_python import Data

# Solvers


# Plotters
from OrbitPlotter import OrbitPlotter


def array_to_latex_matrix(arr, signif=10):
    """
    Convert a NumPy array to a LaTeX matrix with specified significant digits.

    Parameters:
    array (numpy.ndarray): The input NumPy array.
    significant_digits (int): Number of significant digits to format the numbers.

    Returns:
    str: A string representing the LaTeX matrix.
    """

    # Format the array into a LaTeX-compatible string
    formatter = f"{{:.{signif}g}}"
    rows = []
    for row in arr:
        formatted_row = " & ".join([formatter.format(num) for num in row])
        rows.append(formatted_row)
    matrix_body = " \\\\\n".join(rows)

    # Wrap in LaTeX matrix environment
    latex_matrix = f"\\begin{{bmatrix}}\n{matrix_body}\n\\end{{bmatrix}}"
    return latex_matrix


mu = 3.986e5
omg_e = 7.292115e-5  # rad/s


@jit
def U(r):
    """Optimized potential energy calculation"""
    return mu / norm(r)


@jit
def f(x):
    """Optimized state derivative calculation"""
    r = x[0:3]
    v = x[3:6]
    r_norm = norm(r)
    omg = jnp.array([0, 0, omg_e])
    # Compute acceleration directly instead of using grad(U)
    acc = -mu * r / (r_norm**3)
    acc_p = acc - 2 * jnp.cross(omg, v) - jnp.cross(omg, jnp.cross(omg, r))
    return jnp.array([x[3], x[4], x[5], acc_p[0], acc_p[1], acc_p[2]])


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

    # Derivatives    
    x_dot = f
    F_x = lambda x: jax.jacfwd(f)(x)
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

    solution_B = solve_ivp(g_dot_B, (data.t[0], data.t[1]), g_0_B, method="RK45", t_eval=[data.t[0], data.t[1]], atol=1e-6)
    print("\\bar x_1 = ", array_to_latex_matrix(jnp.reshape(solution_B.y[0:6, 1],[6,1]), 6))
    print("\\phi_{1,0} =", array_to_latex_matrix(jnp.reshape(solution_B.y[6:42, 1],[6,6]), 10))
    # print(array_to_latex_matrix(jnp.round(jnp.reshape(solution_B.y[42:48, 1],[6,1]), 10)))
    # fmt: on

    # Part C
    # Functions
    h = lambda x, u: norm(x[0:3] - u, axis=0).reshape(-1, 1)
    # Derivatives
    g_dot = (
        lambda t, g: jnp.concat(  # No sensitivity matrix
            [
                x_dot(g[0:6]),
                jnp.reshape(phi_dot(g[0:6], jnp.reshape(g[6:42], [6, 6])), [36]),
            ]
        )
    )
    H_xn = lambda x, u: jnp.squeeze(jacobian(h)(x, u))

    # Initial conditions
    x_bar = list(
        [
            jnp.array(
                [data.rx[0], data.ry[0], data.rz[0], data.vx[0], data.vy[0], data.vz[0]]
            ).reshape(-1, 1)
        ]
    )

    z_bar = jnp.array(data.CA_range).reshape(100, -1, 1)
    z_bar = [z_bar[i][~jnp.all(z_bar[i] == 0, axis=1)] for i in range(z_bar.shape[0])]
    # print(z_bar[0][0])

    r_gps = jnp.array(
        [jnp.column_stack([data.rx_gps, data.ry_gps, data.rz_gps])]
    ).reshape(100, 3, -1)
    r_gps = [r_gps[i][:, ~np.all(r_gps[i] == 0, axis=0)] for i in range(r_gps.shape[0])]
    # print(r_gps[0][:,0])
    sigma_a = 0.0 #1 # these look like random guesses but i tried 5 orders of magnitude. This is honestly just the best one  i could find
    sigma_b = 0.00005 #1
    Q = jnp.diag(jnp.array([*[pow(sigma_a,2)]*3, *[pow(sigma_b, 2)]*3]))
    P = list([jnp.array(P_xx)+Q])
    R = list([0.009 * jnp.identity(len(z_bar[0]))])

    dz_bar = list([z_bar[0] - h(x_bar[0], r_gps[0])])
    H = list([H_xn(x_bar[0], r_gps[0])])
    K = list([P[0] @ H[0].T @ inv(H[0] @ P[0] @ H[0].T + R[0])])
    x_hat = list([x_bar[0] + K[0] @ dz_bar[0]])
    P_hat = list([(jnp.identity(6) - K[0] @ H[0]) @ P[0]])

    # Propagation loop
    for i in range(len(data.t) - 1):
        # for i in range(10):
        # print("Time: ", i)

        phi_ii = jnp.identity(6)
        g_i = jnp.concat([x_hat[i].reshape([6]), jnp.reshape(phi_ii, [36])])
        g_ip1 = solve_ivp(
            g_dot,
            (data.t[i], data.t[i + 1]),
            g_i,
            method="RK45",
            t_eval=[data.t[i + 1]],
            atol=1e-3,
        )
        x_bar.append(jnp.array([g_ip1.y[0:6]]).reshape([6, 1]))
        phi_ip1i = jnp.array(g_ip1.y[6:]).reshape([6, 6])

        P.append(phi_ip1i @ P_hat[i] @ phi_ip1i.T + Q)

        # Apply corrections
        # So nothing

        # Update with observations
        R.append(0.009 * jnp.identity(len(z_bar[i + 1])))
        H.append(H_xn(x_bar[i + 1], r_gps[i + 1]))
        dz_bar.append(z_bar[i + 1] - h(x_bar[i + 1], r_gps[i + 1]))
        K.append(
            P[i + 1] @ H[i + 1].T @ inv(H[i + 1] @ P[i + 1] @ H[i + 1].T + R[i + 1])
        )
        x_hat.append(x_bar[i + 1] + K[i + 1] @ dz_bar[i + 1])
        P_hat.append((jnp.identity(6) - K[i + 1] @ H[i + 1]) @ P[i + 1])

    orbit_pltr.add_orbit(
        [np.array(jnp.squeeze(x[:-3])) for x in x_hat],
        name="Extended kalman sol",
        color="green",
    )
    # orbit_pltr.plot()

    from ResidualPlotter import ResidualPlotter

    residual = ResidualPlotter()
    residual.add_line_plot(
        data.t - data.t[0],
        1000
        * np.linalg.norm(
            ref_sol[:, :-3] - [np.array(jnp.squeeze(x[:-3])) for x in x_hat], axis=1
        ),
        name="Position difference",
        color="blue",
    )
    residual.add_line_plot(
        data.t - data.t[0],
        1000
        * np.linalg.norm(
            ref_sol[:, -3:] - [np.array(jnp.squeeze(x[-3:])) for x in x_hat], axis=1
        ),
        name="Velocity difference",
        color="green",
        secondary_y=True,
    )
    
    
    # print(
    #     "Deltas epoch 5 - Position: ",
    #     f"{1000 * norm(ref_sol[5][0:3] - x_hat[5][0:3].reshape(3)):.10f}",
    #     " [m],",
    #     "Velocity:",
    #     f"{1000 * norm(ref_sol[5][3:6] - x_hat[5][3:6].reshape(3)):.10f}",
    #     " [m/s]",
    # )
    # print(
    #     "Deltas epoch 9 - Position: ",
    #     f"{1000 * norm(ref_sol[9][0:3] - x_hat[9][0:3].reshape(3)):.10f}",
    #     " [m],",
    #     "Velocity:",
    #     f"{1000 * norm(ref_sol[9][3:6] - x_hat[9][3:6].reshape(3)):.10f}",
    #     " [m/s]",
    # )
    # print(
    #     "Deltas epoch 19 - Position: ",
    #     f"{1000 * norm(ref_sol[19][0:3] - x_hat[19][0:3].reshape(3)):.10f}",
    #     " [m],",
    #     "Velocity:",
    #     f"{1000 * norm(ref_sol[19][3:6] - x_hat[19][3:6].reshape(3)):.10f}",
    #     " [m/s]",
    # )
    # print(
    #     "Deltas epoch 29 - Position: ",
    #     f"{1000 * norm(ref_sol[29][0:3] - x_hat[29][0:3].reshape(3)):.10f}",
    #     " [m],",
    #     "Velocity:",
    #     f"{1000 * norm(ref_sol[29][3:6] - x_hat[29][3:6].reshape(3)):.10f}",
    #     " [m/s]",
    # )

    print("\hat x_10 = ", array_to_latex_matrix(x_hat[9].reshape([1,6]), 6))
    print("\hat x_20 = ", array_to_latex_matrix(x_hat[19].reshape([1,6]), 6))
    print("\hat x_30 = ", array_to_latex_matrix(x_hat[29].reshape([1,6]), 6))

    P = jnp.stack(P)
    from StatisticsPlotter import StatisticsPlotter
    stat_pltr = StatisticsPlotter()
    stat_pltr.add_line_plot(data.t - data.t[0], jnp.sqrt(P[:,0,0])*1000, name="Sigma x", color="red")
    stat_pltr.add_line_plot(data.t - data.t[0], jnp.sqrt(P[:,1,1])*1000, name="Sigma y", color="green")
    stat_pltr.add_line_plot(data.t - data.t[0], jnp.sqrt(P[:,2,2])*1000, name="Sigma z", color="blue")
    stat_pltr.add_line_plot(data.t - data.t[0], [len(x) for x in z_bar], name="measurements", color="pink")
    
    stat_pltr.add_line_plot(data.t - data.t[0], jnp.sqrt(P[:,3,3])*1000, name="Sigma vx", color="red", secondary_y=True)
    stat_pltr.add_line_plot(data.t - data.t[0], jnp.sqrt(P[:,4,4])*1000, name="Sigma vy", color="green", secondary_y=True)
    stat_pltr.add_line_plot(data.t - data.t[0], jnp.sqrt(P[:,5,5])*1000, name="Sigma vz", color="blue", secondary_y=True)
    sig_r = norm(jnp.stack([jnp.sqrt(P[:,0,0])*1000, jnp.sqrt(P[:,1,1])*1000, jnp.sqrt(P[:,2,2])*1000]), axis=0)
    sig_v = norm(jnp.stack([jnp.sqrt(P[:,3,3])*1000, jnp.sqrt(P[:,4,4])*1000, jnp.sqrt(P[:,5,5])*1000]), axis=0)
    stat_pltr.add_line_plot(data.t - data.t[0], sig_r, name="Sigma r", color="orange")
    stat_pltr.add_line_plot(data.t - data.t[0], sig_v, name="Sigma v", color="purple", secondary_y=True)
    
    # residual_pltr.add_line_plot(data.t - data.t[0], sig_r, name="Sigma r", color="orange", secondary_y=True)
    # residual_pltr.add_line_plot(data.t - data.t[0], sig_v, name="Sigma v", color="purple")
    
    residual_pltr.plot()
    stat_pltr.plot()
    
    # Part G
    #For task f, calculate for every epoch the RMS of the observation residuals 
    # $\bar e_k = \Delta \bar z_k - H_k \Delta \hat x_k$ where $\Delta \hat x_k = K \Delta \bar z_k$ in the Kalman filter, 
    # compare the size to the position errors δr, and interpret the comparison. (10 points)
    
    dx_hat = []
    e_bar = []
    rms_e_bar = []
    for i in range(len(data.t)):
        dx_hat.append(K[i] @ dz_bar[i])
        e_bar.append(dz_bar[i] - H[i] @ dx_hat[i])
        rms_e_bar.append(
            jnp.sqrt(
                jnp.sum(jnp.pow(1000*e_bar[i], 2)/(e_bar[i].shape[0]))
            )
        )
        
    rms_pltr = ResidualPlotter()
    rms_pltr.add_line_plot(
        data.t - data.t[0],
        rms_e_bar,
        "RMS of e hat",
        "green"
    )
    rms_pltr.add_line_plot(
        data.t - data.t[0],
        1000
        * np.linalg.norm(
            ref_sol[:, :-3] - [np.array(jnp.squeeze(x[:-3])) for x in x_hat], axis=1
        ),
        name="Position difference",
        color="blue",
    )
    rms_pltr.plot()    
    

if __name__ == "__main__":
    main()
    # import timeit
    # num = 20
    # time_f = timeit.timeit(lambda: main(), number=num)
    # print(f"Time for main(): {time_f/num:.6f} seconds")
