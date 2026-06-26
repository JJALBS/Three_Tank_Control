import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
import torch
import os

from sensor_control import activate_sensor
from models.PID import PID
from models.recurrentPPO import RecurrentPPO
from models.RL import DQN
from models.SAC import SoftActorCritic
from models.MPC import ModelPredictiveControl
from models.MPC import KalmanFilter

from AILevelingRL.PyArduino import PyArduino
from analysis import analysis

from util.util import delete_spike

"""
mv = pump speed
cv = liquid level
sp = set point
"""

CONFIG_FILE_MAP = {
    "P": "P.yaml",
    "PI": "PI.yaml",
    "PID": "PID.yaml",
    "DQN": "DQN.yaml",
    "RPPO": "recurrentPPO.yaml",
    "SAC": "SAC.yaml",
    "MPC": "MPC.yaml",
}

MODELPARAM_FILE_MAP = {
    "DQN": "DQN.pt",
    "RPPO": "recurrentPPO.pt",
    "SAC": "SAC.pt",
}

MAX_H = 8.5  # This is the max height of the tank in cm. This will be used to normalize the input for RL agents.

with open(r"parameters\steady_state_param.pickle", "rb") as f:
    steady_param = pickle.load(f)

with open(r"parameters\state_space_param.pickle", "rb") as f:
    statespaceparam = pickle.load(f)
A = statespaceparam["A"]
B = statespaceparam["B"]
C = statespaceparam["C"]
I = np.eye(2)
A2 = A @ A
A3 = A2 @ A
A4 = A3 @ A


def call_controller(config, args_controller, init_guess, set_point):
    if config["controller_mode"] == "PID":
        if config["controller"] is None:
            raise ValueError(
                f"Missing 'controller' configuration in config for {args_controller} controller"
            )
        controller = PID(**config["controller"])

    elif config["controller_mode"] == "RL":
        model_param_path = rf"parameters\{MODELPARAM_FILE_MAP[args_controller]}"
        if args_controller == "DQN":
            controller = DQN(config["network_arg"], config["lr"], config["replay_capa"])
            controller.Q_func.load_state_dict(torch.load(model_param_path))
            controller.Q_func.eval()
        elif args_controller == "RPPO":
            controller = RecurrentPPO(
                config["network_arg"], config["lr"], config["pum_max_speed"]
            )
            controller.network.load_state_dict(torch.load(model_param_path))
            controller.network.eval()
        elif args_controller == "SAC":
            controller = SoftActorCritic(
                config["v_net_arg"],
                config["q_net_arg"],
                config["actor_net_arg"],
                config["replay_capa"],
            )
            controller.actor.load_state_dict(torch.load(model_param_path))
            controller.actor.eval()
        else:
            raise TypeError(
                "the config file state its controller_mode is RL, but the args.controller does not match with any of the RL controller types in [DQN, rPPO, SAC]"
            )
        # controller.load_state_dict(torch.load(model_param_path))

    elif config["controller_mode"] == "MPC":
        controller = ModelPredictiveControl(
            init_guess=init_guess, set_point=set_point, **config["controller"]
        )

    else:
        raise TypeError(
            "the controller mode from the config file does not match any of the standard controller mode explored in this experiment."
        )

    return controller

 
def controller_action_selection(controller, set_point, estimated_cv, measured_cv, dt):
    if isinstance(controller, PID):
        action = controller.final_control(set_point, estimated_cv, dt)
    elif isinstance(controller, DQN):
        state = [estimated_cv / MAX_H, set_point / MAX_H]
        action = controller.greedy_action(state)
    elif isinstance(controller, RecurrentPPO):
        state = [estimated_cv / MAX_H, set_point / MAX_H]
        action = controller.deterministic_action(state)
    elif isinstance(controller, SoftActorCritic):
        state = [estimated_cv / MAX_H, set_point / MAX_H]
        action = controller.choose_action(state, distributional=False)
    elif isinstance(controller, ModelPredictiveControl):
        action, U_opt, X_opt = controller.control(
            dt=3, u_min=0.0, u_max=36.0
        )
        controller.update_state(measured_cv, action)
    else:
        raise ValueError(
            f"for controller_action_selection function, the input controller does not match with any of the controller class defined in this experiment."
        )
    return action


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--experiment_horizon",
        type=float,
        default=15.0,
        help="total time for the experiment to proceed in minutes.",
    )
    parser.add_argument(
        "-sp", "--set_point", type=float, default=4.5, help="set point in cm."
    )
    parser.add_argument(
        "-c",
        "--controller",
        type=str.upper,
        required=True,
        choices=["P", "PI", "PID", "DQN", "RPPO", "SAC", "MPC"],
        help="string name of the controller to test: [P, PI, PID, DQN, rPPO, SAC, MPC]. (rPPO = short for recurrentPPO)",
    )
    parser.add_argument(
        "-i",
        "--initial_state",  # this will be used as a initial guess for state estimation in MPC
        type=float,
        nargs=2,
        required=True,
        help="initial state of the two tank at the begining of the experiment (first input being the first tank and the second input being the second tank).",
    )
    parser.add_argument(
        "-dt",
        "--time_interval",
        type=float,
        default=3.0,
        help="time interval between each control decision in seconds (ie. inverse of sampling rate).",
    )
    args = parser.parse_args()

    Ad = (
        I
        + args.time_interval * A
        + (args.time_interval**2 / 2) * A2
        + (args.time_interval**3 / 6) * A3
        + (args.time_interval**4 / 24) * A4
    )
    Bd = (
        args.time_interval * I
        + (args.time_interval**2 / 2) * A
        + (args.time_interval**3 / 6) * A2
        + (args.time_interval**4 / 24) * A3
        + (args.time_interval**5 / 120) * A4
    ) @ B

    state_estimator = KalmanFilter(
        init_guess=args.initial_state,
        measurement_covar=np.array([[0.246]]),
        A=Ad,
        B=Bd,
        C=C,
        state_noise_covar=np.diag([2e-2, 2e-2]),
        estim_err_covar=np.diag([0.5**2, 0.5**2]),
    )

    cv_list = []
    error_list = []
    mv_list = []
    time_list = []
    estimated_h1_list, estimated_h2_list = [], []
    steady_state_pump_speed = np.sqrt(
        (args.set_point - steady_param["B"]) / steady_param["A"]
    )

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(14, 8),
        dpi=100,
    )
    fig.subplots_adjust(right=0.88, hspace=0.18)
    scatter_raw_h2 = ax1.scatter(
        [],
        [],
        color="C0",
        s=10,
        alpha=0.4,
        label="raw measurement (h2)",
        zorder=3,
    )
    (line_estimated_h1,) = ax1.plot(
        time_list, estimated_h1_list, c="C1", label="estimated h1"
    )
    (line_estimated_h2,) = ax1.plot(
        time_list, estimated_h2_list, c="C0", label="estimated h2"
    )
    ax1.set_ylabel("Liquid level, cv [cm]")
    ax1.axhline(y=args.set_point, ls="--", c="r", label="set point")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # (line_mv,) = ax2.plot(time_list, mv_list, c="tab:blue", label="Pump speed")
    (line_mv,) = ax2.step(time_list, mv_list, c="tab:blue", label="Pump speed")
    ax2.set_ylabel("Pump speed, mv [rpm]")
    ax2.tick_params(axis="y")
    ax2.axhline(
        y=steady_state_pump_speed, ls="--", c="orange", label="Steady state pump speed"
    )
    ax3 = ax2.twinx()
    (line_err,) = ax3.plot(time_list, error_list, color="green", label="Error")
    ax3.set_ylabel("Error [cm]")
    ax3.tick_params(axis="y")
    ax3.axhline(y=0, lw=0.5, c="k")

    handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
    handles_ax3, labels_ax3 = ax3.get_legend_handles_labels()

    ax2.legend(
        handles_ax2 + handles_ax3,
        labels_ax2 + labels_ax3,
        loc="upper right",
    )

    ax1.set_ylim(0, 9)
    ax2.set_ylim(0, 37)
    ax3.set_ylim(-9, 9)

    num_ticks = 5
    ticks = np.linspace(-1, 1, num_ticks)
    zero_ticks = np.linspace(0, 1, num_ticks)
    ax2.set_yticks(zero_ticks * 36)
    ax3.set_yticks(ticks * 8)

    ax2.set_xlabel("Time [seconds]")
    ax2.set_xlim(
        0, args.experiment_horizon * 60 * 1.1
    )  # Although we have set the inference to occur every some period of time, due to the latency within the process, it usually may take longer than expected
    ax2.grid(True, alpha=0.3)

    plt.ion()
    plt.show(block=False)

    config_path = rf"config\{CONFIG_FILE_MAP[args.controller]}"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    controller = call_controller(
        config, args.controller, args.initial_state, args.set_point
    )

    sensor_params = config.get("sensor")
    if sensor_params is None:
        raise ValueError("Missing 'sensor' configuration in config")
    sensor = activate_sensor(YOLO_path=r"parameters\YOLO.pt", **sensor_params)

    UPPER_VALVE_PIN = config["UPPER_VALVE_PIN"]
    LOWER_VALVE_PIN = config["LOWER_VALVE_PIN"]
    PUMP_PIN = config["PUMP_PIN"]

    # For the control scenario purpose, we open both upper and lower valve for entire time
    # Initialise PyArduino
    pa = PyArduino("minima")
    # Open both valve
    pa.run_digital_write(UPPER_VALVE_PIN, True)
    pa.run_digital_write(LOWER_VALVE_PIN, True)
    pa.run_digital_write(7, False) #This is true/flase logic to operates the disturbance pump. The pump speed cannot be controlled.

    # =================================================================================================
    start_time = time.perf_counter()
    prev_time = 0
    prev_cv = None
    try:
        for cv in sensor:
            now = time.perf_counter()

            proceed = delete_spike(prev_cv, cv, threshold=0.1)

            if proceed:

                if prev_cv == None:
                    state = state_estimator.posteriori_state_estim
                else:
                    state = state_estimator.state_estimation(new_measurement=cv, u=mv)

                cv_list.append(cv)
                estimated_h1_list.append(state[0].item())
                estimated_h2_list.append(state[1].item())

                error = args.set_point - state[1].item()

                mv = controller_action_selection(
                    controller=controller,
                    set_point=args.set_point,
                    estimated_cv=state[1].item(),
                    measured_cv=cv,
                    dt=now - max(start_time, prev_time),
                )

                prev_time = now
                error_list.append(error)
                time_list.append(now - start_time)
                pa.run_pump_speed(target_speed=mv)
                mv_list.append(mv)

                line_mv.set_data(time_list, mv_list)
                scatter_raw_h2.set_offsets(np.column_stack((time_list, cv_list)))
                line_err.set_data(time_list, error_list)
                line_estimated_h1.set_data(time_list, estimated_h1_list)
                line_estimated_h2.set_data(time_list, estimated_h2_list)

                fig.canvas.draw()
                fig.canvas.flush_events()

                if time.perf_counter() - start_time >= args.experiment_horizon * 60:
                    break

            prev_cv = cv

    finally:
        pa.run_pump_speed(target_speed=0)
        sensor.close()
        plt.close(fig)

    ie, iae, ise, itae = analysis(error_list, time_list)

    result_data = {
        "sp": args.set_point,
        "init_state": args.initial_state,
        "mv": mv_list,
        "err": error_list,
        "raw measurement": cv_list,
        "estimated h1": estimated_h1_list,
        "estimated h2": estimated_h2_list,
        "time": time_list,
        "ie_list": np.insert(ie, 0, 0),
        "ie_final": ie[-1],
        "iae_list": np.insert(iae, 0, 0),
        "iae_final": iae[-1],
        "ise_list": np.insert(ise, 0, 0),
        "ise_final": ise[-1],
        "itae_list": np.insert(itae, 0, 0),
        "itae_final": itae[-1],
    }

    os.makedirs(r"data\control_result\abnormal_initial_state", exist_ok=True)
    for i in range(0, 1000):
        filename = rf"data\control_result\abnormal_initial_state\{config["controller_type"]}_{i}.pickle"
        if not os.path.exists(filename):
            break

    with open(filename, "wb") as f:
        pickle.dump(result_data, f)

    print("Control test result saved as:", filename)

    pa.run_digital_write(UPPER_VALVE_PIN, False)
    pa.run_digital_write(LOWER_VALVE_PIN, False)
    pa.run_digital_write(7, True)