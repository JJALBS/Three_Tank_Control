import casadi as ca
import numpy as np


class StateSpaceModel:
    def __init__(self, pred_x0: bool):
        self.pred_x0 = pred_x0

    def multi_exp_param_estimation(self, cv, time, u, silent: bool = False):
        opti = ca.Opti()
        n_traj = cv.shape[0]

        # declaring decision variables
        # free parameters only
        a11 = opti.variable()
        a21 = opti.variable()
        a22 = opti.variable()
        b1 = opti.variable()

        # known structure
        A = ca.vertcat(ca.horzcat(a11, 0), ca.horzcat(a21, a22))
        B = ca.vertcat(b1, 0)
        C = ca.DM([[0, 1]])
        D = ca.DM([0])

        if self.pred_x0:
            X0 = opti.variable(2, n_traj)

        J_data = 0

        for i in range(n_traj):
            uk = float(u[i])

            if self.pred_x0:
                xk = X0[:, i]
            else:
                xk = ca.DM([[float(cv[i][0])], [float(cv[i][0])]])

            n_i = cv[i].shape[0]

            for j in range(n_i):
                yk = C @ xk + D * uk
                err = yk - float(cv[i][j])
                J_data += ca.sumsqr(err)

                if j < n_i - 1:
                    dt_ij = float(time[i][j + 1] - time[i][j])
                    Ad_ij, Bd_ij = self.discretize(A, B, dt_ij)
                    xk = Ad_ij @ xk + Bd_ij * uk

        lam = 1e-4
        J_reg = lam * (ca.sumsqr(A) + ca.sumsqr(B))

        opti.minimize(J_data + J_reg)

        opti.subject_to(opti.bounded(-2, ca.vec(A), 2))
        opti.subject_to(opti.bounded(-2, ca.vec(B), 2))

        if self.pred_x0:
            opti.subject_to(opti.bounded(0, ca.vec(X0), 8.5))

        p_opts = {"expand": True}

        s_opts = {"max_iter": 50000, "print_level": 0 if silent else 5, "tol": 1e-3}

        opti.solver("ipopt", p_opts, s_opts)
        sol = opti.solve()

        A_hat = np.array(sol.value(A))
        B_hat = np.array(sol.value(B))
        C_hat = np.array(sol.value(C))
        D_hat = np.array(sol.value(D))

        if self.pred_x0:
            X0_hat = np.array(sol.value(X0))
            return A_hat, B_hat, C_hat, D_hat, X0_hat
        return A_hat, B_hat, C_hat, D_hat

    def discretize(self, A, B, dt):
        """
        Fast discrete-time matrices for one interval dt using RK4/Taylor-4 form.
        Works with CasADi symbolic variables and variable dt.
        """
        I = ca.DM.eye(2)
        h = dt

        A2 = A @ A
        A3 = A2 @ A
        A4 = A3 @ A

        Ad = I + h * A + (h**2 / 2.0) * A2 + (h**3 / 6.0) * A3 + (h**4 / 24.0) * A4
        Bd = (
            h * I
            + (h**2 / 2.0) * A
            + (h**3 / 6.0) * A2
            + (h**4 / 24.0) * A3
            + (h**5 / 120.0) * A4
        ) @ B

        return Ad, Bd

    def step_pred(self, A, B, C, D, xk, u, dt, return_discrete=False):
        """
        One-step prediction with variable dt using fast RK4-based discretization.
        """
        A = np.asarray(A, dtype=float).reshape(2, 2)
        B = np.asarray(B, dtype=float).reshape(2, 1)
        C = np.asarray(C, dtype=float).reshape(1, 2)
        xk = np.asarray(xk, dtype=float).reshape(2, 1)
        D = float(np.asarray(D).squeeze())
        u = float(u)
        dt = float(dt)

        yk = C @ xk + D * u

        A2 = A @ A
        A3 = A2 @ A
        A4 = A3 @ A
        I = np.eye(2)

        Ad = I + dt * A + (dt**2 / 2.0) * A2 + (dt**3 / 6.0) * A3 + (dt**4 / 24.0) * A4
        Bd = (
            dt * I
            + (dt**2 / 2.0) * A
            + (dt**3 / 6.0) * A2
            + (dt**4 / 24.0) * A3
            + (dt**5 / 120.0) * A4
        ) @ B

        x_k1 = Ad @ xk + Bd * u

        if return_discrete:
            return x_k1, yk, Ad, Bd
        return x_k1, yk
