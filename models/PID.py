import time
import os
import numpy as np
from scipy.integrate import cumulative_trapezoid
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAM_PATH = os.path.join(BASE_DIR, r"..\parameters\steady_state_param.pickle")

with open(PARAM_PATH, "rb") as f:
    param = pickle.load(f)


class PID:
    def __init__(
        self,
        kc: float,
        P: bool,
        I: bool,
        D: bool,
        u_min: float,
        u_max: float,
        taui: float | None = None,
        taud: float | None = None,
    ):
        self.kc = kc
        self.P = P
        self.I = I
        self.taui = taui
        self.D = D
        self.taud = taud
        self.u_min = u_min
        self.u_max = u_max

        self.prev_error = 0
        self.prev_integral = 0

        self.prev_cv = 0

        if self.I and (self.taui is None or self.taui <= 0):
            raise ValueError("taui must be > 0 when I control is enabled.")
        if self.D and (self.taud is None or self.taud <= 0):
            raise ValueError("taud must be > 0 when D control is enabled.")

    def error(self, sp, cv):
        return sp - cv

    def P_control(self, sp, cv):
        return self.kc * self.error(sp, cv)

    def I_control(self, sp, cv, dt):
        new_error = self.error(sp, cv)
        self.new_integral = 0.5 * (self.prev_error + new_error) * dt

        integral = self.prev_integral + self.new_integral

        self.prev_integral = integral
        self.prev_error = new_error

        return self.kc * integral / self.taui

    def D_control(self, cv, dt):
        derivative = (cv - self.prev_cv) / dt

        self.prev_cv = cv

        return self.kc * self.taud * derivative

    def Steady_State_Pump_Speed(self, sp):
        if sp < param["B"]:
            raise ValueError(
                f'set point at {sp} and param["B"] at {param["B"]}: set point too low to calculate steady state pump speed.'
            )
        return np.sqrt((sp - param["B"]) / param["A"])

    def final_control(self, sp, cv, dt):
        if cv is None:
            # This is to allow the controller to handle when it encounter cv = None.
            # This is possible case as approx_h from sensor_control.py will be None when sensor cannot detect the water tank.
            mv_execute = 0
        else:
            mv = 0.0

            if self.P:
                mv += self.P_control(sp, cv)
            if self.I:
                mv += self.I_control(sp, cv, dt)
            if self.D:
                mv -= self.D_control(cv, dt)

            mv += self.Steady_State_Pump_Speed(sp)

            mv_execute = np.clip(mv, self.u_min, self.u_max)

            # logic to prevent integral windup. why: 1) max and min exist for u, and 2) slow reaction of water tank
            if self.I:
                saturated = (mv > self.u_max) or (mv < self.u_min)
                if saturated:
                    self.prev_integral -= self.new_integral

            if self.I or self.D:
                self.prev_t = time.time()

        return float(mv_execute)
