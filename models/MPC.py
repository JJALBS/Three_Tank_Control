import casadi as ca
import numpy as np
from collections import deque
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PARAM_DIR = BASE_DIR / "parameters"

with open(PARAM_DIR / "state_space_param.pickle", "rb") as f:
    statespaceparam = pickle.load(f)

A = statespaceparam["A"]
B = statespaceparam["B"]
C = statespaceparam["C"]

with open(PARAM_DIR / "steady_state_param.pickle", "rb") as f:
    param = pickle.load(f)


class environment:
    def __init__(self):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)

        if self.B.ndim == 1:
            self.B = self.B.reshape(-1, 1)

        if self.C.ndim == 1:
            self.C = self.C.reshape(1, -1)

        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        self.ny = self.C.shape[0]

        self.I = np.eye(self.nx)

        self.A2 = self.A @ self.A
        self.A3 = self.A2 @ self.A
        self.A4 = self.A3 @ self.A

        dt = 3
        self.Ad = (
            self.I
            + dt * self.A
            + (dt**2 / 2) * self.A2
            + (dt**3 / 6) * self.A3
            + (dt**4 / 24) * self.A4
        )

        self.Bd = (
            dt * self.I
            + (dt**2 / 2) * self.A
            + (dt**3 / 6) * self.A2
            + (dt**4 / 24) * self.A3
            + (dt**5 / 120) * self.A4
        ) @ self.B

        self.x_min = np.zeros((self.nx, 1))
        self.x_max = 8.5 * np.ones((self.nx, 1))

    def step(self, x_t, u_t, dt=3):
        """
        NumPy-based simulation step.
        Use this for simulating the real/nominal environment.
        Do not use this directly inside CasADi optimization.
        """
        x_t = np.asarray(x_t, dtype=float).reshape(self.nx, 1)
        u_t = np.asarray(u_t, dtype=float).reshape(self.nu, 1)

        x_t = np.clip(x_t, self.x_min, self.x_max)
        x_t1 = self.Ad @ x_t + self.Bd @ u_t
        x_t1 = np.clip(x_t1, self.x_min, self.x_max)

        y_t1 = self.C @ x_t1

        return x_t1, y_t1

    def step_casadi(self, x_t, u_t, dt=3):
        """
        CasADi-compatible prediction step.
        Use this inside MPC optimization.

        Important:
        Do not clip states here.
        State bounds should be imposed as optimization constraints.
        """
        Ad = ca.DM(self.Ad)
        Bd = ca.DM(self.Bd)
        C_ca = ca.DM(self.C)

        x_t = ca.reshape(x_t, self.nx, 1)
        u_t = ca.reshape(u_t, self.nu, 1)

        x_t1 = Ad @ x_t + Bd @ u_t
        y_t = C_ca @ x_t

        return x_t1, y_t

    def trajectory_pred(
        self, x_0, u_sequence, dt_sequence=None, state_estimation=False
    ):
        y_sequence = []
        x_t = x_0

        if dt_sequence is None:
            dt_sequence = np.ones(len(u_sequence)) * 3

        for i in range(len(u_sequence)):
            x_t, y_t = self.step(x_t, u_sequence[i], dt_sequence[i])
            y_sequence.append(y_t)

        y_t = self.C @ x_t
        y_sequence.append(y_t)

        if state_estimation:
            return y_sequence, x_t
        else:
            return y_sequence


class KalmanFilter:
    def __init__(
        self,
        init_guess,
        measurement_covar,
        A,
        B,
        C,
        state_noise_covar=None,
        estim_err_covar=None,
    ):
        self.posteriori_state_estim = np.asarray(init_guess, dtype=float).reshape(-1, 1)

        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        if self.B.ndim == 1:
            self.B = self.B.reshape(-1, 1)
        if self.C.ndim == 1:
            self.C = self.C.reshape(1, -1)

        n = self.A.shape[0]
        m = self.C.shape[0]

        self.measurement_covar = np.asarray(measurement_covar, dtype=float).reshape(
            m, m
        )

        if state_noise_covar is None:
            self.state_noise_covar = np.zeros((n, n))
        else:
            self.state_noise_covar = np.asarray(state_noise_covar, dtype=float).reshape(
                n, n
            )

        if estim_err_covar is None:
            # Avoid np.inf. Use a large finite covariance instead.
            self.posteriori_estim_err_covar = np.eye(n) * 1e6
        else:
            self.posteriori_estim_err_covar = np.asarray(
                estim_err_covar, dtype=float
            ).reshape(n, n)

        self.u_storage = deque(
            [np.asarray([], dtype=float).reshape(-1, 1)],
            maxlen=2,
        )

        # Diagnostics
        self.innovation_history = []
        self.S_history = []
        self.nis_history = []
        self.P_is_spd_history = []

    @staticmethod
    def is_symmetric_positive_definite(P, symmetry_tol=1e-6):
        P = np.asarray(P)

        # Check square
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            return False

        # Check symmetry
        if not np.allclose(P, P.T, atol=symmetry_tol):
            return False

        # Check positive definiteness
        try:
            np.linalg.cholesky(P)
            return True
        except np.linalg.LinAlgError:
            return False

    def state_estimation(self, new_measurement, u):
        y = np.asarray(new_measurement, dtype=float).reshape(self.C.shape[0], 1)
        u = np.asarray(u, dtype=float).reshape(self.B.shape[1], 1)

        # 1. A priori prediction
        self.priori_state_estim = self.A @ self.posteriori_state_estim + self.B @ u
        self.priori_estim_err_covar = (
            self.A @ self.posteriori_estim_err_covar @ self.A.T + self.state_noise_covar
        )

        # 2. Innovation
        innovation = y - self.C @ self.priori_state_estim
        S = self.C @ self.priori_estim_err_covar @ self.C.T + self.measurement_covar
        asymmetry = np.linalg.norm(S - S.T)
        if asymmetry > 1e-8:
            print("Warning: S is noticeably asymmetric:", asymmetry)
        S = 0.5 * (S + S.T)
        # S is the innovation covariance matrix. Analytically, it should be symmetric so we took the numerical symmetrization step here.

        # 3. Kalman gain
        # self.gain = self.priori_estim_err_covar @ self.C.T @ np.linalg.inv(S). Computation of inverse matrix is slow and fragile. In fact, solving linear algebra is more stable than computing inverse matrix.
        PCt = self.priori_estim_err_covar @ self.C.T
        self.gain = np.linalg.solve(S.T, PCt.T).T

        # 4. A posteriori state update
        self.posteriori_state_estim = self.priori_state_estim + self.gain @ innovation

        # 5. A posteriori estimation-error covariance update
        I = np.eye(self.posteriori_estim_err_covar.shape[0])
        I_KC = I - self.gain @ self.C
        # initially use the computationally simple version
        self.posteriori_estim_err_covar = I_KC @ self.priori_estim_err_covar
        # check if the covariance is positive definite. if failed, recalculate estimation-error covariance using Joseph stabilized form
        if self.is_symmetric_positive_definite(self.posteriori_estim_err_covar):
            pass
        else:
            self.posteriori_estim_err_covar = (
                I_KC @ self.priori_estim_err_covar @ I_KC.T
                + self.gain @ self.measurement_covar @ self.gain.T
            )
        # if not self.is_symmetric_positive_definite(self.posteriori_estim_err_covar):
        # print("Warning: posterior covariance P is not semmetric positive-definite even after Joseph update.")
        self.posteriori_estim_err_covar = 0.5 * (
            self.posteriori_estim_err_covar + self.posteriori_estim_err_covar.T
        )

        # 6. Diagnostics
        L = np.linalg.cholesky(S)
        normalized_innovation = np.linalg.solve(L, innovation)
        nis = (normalized_innovation.T @ normalized_innovation).item()

        self.innovation_history.append(innovation.copy())
        self.S_history.append(S.copy())
        self.nis_history.append(nis)
        self.P_is_spd_history.append(
            self.is_symmetric_positive_definite(self.posteriori_estim_err_covar)
        )

        return self.posteriori_state_estim


class ModelPredictiveControl:
    def __init__(
        self,
        init_guess,
        measurement_covar,
        A,
        B,
        C,
        Q,
        R,
        N,
        set_point,
        state_noise_covar=None,
        estim_err_covar=None,
    ):
        self.kf = KalmanFilter(
            init_guess,
            measurement_covar,
            A,
            B,
            C,
            state_noise_covar,
            estim_err_covar,
        )
        self.Q = Q
        self.R = R
        self.state = init_guess
        self.N = N
        self.set_point = set_point
        self.sys = environment()
        self.steady_state_pump_speed = np.sqrt((set_point - param["B"]) / param["A"])

    def update_state(self, observation, input):
        self.state = self.kf.state_estimation(observation, input)

        # Keep estimated state physically feasible
        self.state = np.clip(
            np.asarray(self.state, dtype=float).reshape(self.sys.nx, 1),
            self.sys.x_min,
            self.sys.x_max,
        )

    def control(
        self,
        dt=3,
        u_min=None,
        u_max=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
    ):
        """
        Multiple-shooting MPC.

        Decision variables:
            X[:, 0], X[:, 1], ..., X[:, N]
            U[:, 0], U[:, 1], ..., U[:, N-1]

        Constraints:
            X[:, 0] = current estimated state
            X[:, k+1] = f(X[:, k], U[:, k])
            optional input/state/output bounds

        Objective:
            sum tracking error + input penalty
        """

        opti = ca.Opti()

        nx = self.sys.nx
        nu = self.sys.nu
        ny = self.sys.ny
        N = self.N

        Q = ca.DM(np.asarray(self.Q, dtype=float))
        R = ca.DM(np.asarray(self.R, dtype=float))

        x0 = np.asarray(self.state, dtype=float).reshape(nx, 1)
        r = np.asarray(self.set_point, dtype=float).reshape(ny, 1)
        r = ca.DM(r)

        # -----------------------------
        # Decision variables
        # -----------------------------
        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)
        U_actual = U + self.steady_state_pump_speed

        # -----------------------------
        # Initial condition constraint
        # -----------------------------
        opti.subject_to(X[:, 0] == ca.DM(x0))

        # -----------------------------
        # Objective
        # -----------------------------
        J = 0

        for k in range(N):
            x_k = X[:, k]
            delta_u_k = U[:, k]
            u_k = U_actual[:, k]

            # Predict one step using the CasADi-compatible environment model
            x_next_pred, y_k = self.sys.step_casadi(x_k, u_k, dt=dt)

            # Multiple-shooting equality constraint
            opti.subject_to(X[:, k + 1] == x_next_pred)

            # Tracking error
            e_y = y_k - r

            # Stage cost
            J += ca.mtimes([e_y.T, Q, e_y]) + ca.mtimes([delta_u_k.T, R, delta_u_k])

            # Optional state constraints
            if x_min is not None:
                opti.subject_to(X[:, k] >= x_min)
            if x_max is not None:
                opti.subject_to(X[:, k] <= x_max)

            # Optional output constraints
            if y_min is not None:
                opti.subject_to(y_k >= y_min)
            if y_max is not None:
                opti.subject_to(y_k <= y_max)

        # Terminal output tracking cost
        y_terminal = ca.DM(self.sys.C) @ X[:, N]
        e_terminal = y_terminal - r
        J += ca.mtimes([e_terminal.T, Q, e_terminal])

        # Optional terminal state constraints
        if x_min is not None:
            opti.subject_to(X[:, N] >= x_min)
        if x_max is not None:
            opti.subject_to(X[:, N] <= x_max)

        opti.minimize(J)

        # -----------------------------
        # Input constraints
        # -----------------------------
        if u_min is not None:
            opti.subject_to(U_actual >= u_min)

        if u_max is not None:
            opti.subject_to(U_actual <= u_max)

        # -----------------------------
        # Initial guesses
        # -----------------------------
        opti.set_initial(X, np.tile(x0, (1, N + 1)))

        try:
            last_u = np.asarray(self.kf.u_storage[-1], dtype=float).reshape(nu, 1)
            opti.set_initial(U_actual, np.tile(last_u, (1, N)))
        except Exception:
            opti.set_initial(U_actual, np.zeros((nu, N)))

        # -----------------------------
        # Solver settings
        # -----------------------------
        opts = {
            "ipopt.print_level": 0,
            "print_time": False,
            "ipopt.max_iter": 500,
        }

        opti.solver("ipopt", opts)

        # -----------------------------
        # Solve NLP
        # -----------------------------
        sol = opti.solve()

        U_opt = np.asarray(sol.value(U), dtype=float).reshape(nu, N)
        X_opt = np.asarray(sol.value(X), dtype=float).reshape(nx, N + 1)

        u0 = U_opt[:, [0]]

        # For SISO system, return scalar control if desired
        if nu == 1:
            u0_to_apply = float(u0.item()) + self.steady_state_pump_speed
        else:
            u0_to_apply = u0 + self.steady_state_pump_speed

        return u0_to_apply, U_opt, X_opt
