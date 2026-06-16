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

with open(r"parameters\steady_state_param.pickle", "rb") as f:
    steady_param = pickle.load(f)


def call_controller(config, args_controller, init_guess, set_point):
    if config["controller_mode"] == "PID":
        if config["controller_params"] is None:
            raise ValueError(
                f"Missing 'controller' configuration in config for {args_controller} controller"
            )
        controller = PID(**config["controller_params"])

    elif config["controller_mode"] == "RL":
        model_param_path = rf"parameters\{MODELPARAM_FILE_MAP[args_controller]}"
        if args_controller == "DQN":
            controller = DQN(config["network_arg"], config["lr"], config["replay_capa"])
        elif args_controller == "RPPO":
            controller = RecurrentPPO(
                config["network_arg"], config["lr"], config["pum_max_speed"]
            )
        elif args_controller == "SAC":
            controller = SoftActorCritic(
                config["v_net_arg"],
                config["q_net_arg"],
                config["actor_net_arg"],
                config["replay_capa"],
            )
        else:
            raise TypeError(
                "the config file state its controller_mode is RL, but the args.controller does not match with any of the RL controller types in [DQN, rPPO, SAC]"
            )
        controller.load_state_dict(torch.load(model_param_path))

    elif config["controller_mode"] == "MPC":
        controller = ModelPredictiveControl(
            init_guess=init_guess, set_point=set_point, **config["controller"]
        )

    else:
        raise TypeError(
            "the controller mode from the config file does not match any of the standard controller mode explored in this experiment."
        )

    return controller


def controller_action_selection(controller, set_point, control_variable, dt):
    if isinstance(controller, PID):
        action = controller.final_control(set_point, control_variable, dt)
    elif isinstance(controller, DQN):
        action = controller.greedy_action(control_variable)
    elif isinstance(controller, RecurrentPPO):
        action = controller.deterministic_action(control_variable)
    elif isinstance(controller, SoftActorCritic):
        action = controller.choose_action(control_variable)
    elif isinstance(controller, ModelPredictiveControl):
        action, U_opt, X_opt = controller.control(
            dt=3, u_min=0.0, u_max=36.0, x_min=0.0, x_max=8.5
        )
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
        default=(0.0, 0.0),
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

    cv_list = []
    error_list = []
    mv_list = []
    time_list = []
    filtered_cv = []
    steady_state_pump_speed = np.sqrt(
        (args.set_point - steady_param["B"]) / steady_param["A"]
    )

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig.subplots_adjust(right=0.78)
    (line_cv,) = ax1.plot(time_list, cv_list, label="liquid livel")
    (line_filter_cv,) = ax1.plot(time_list, filtered_cv, label="moving average")
    ax1.set_ylabel("Liquid level, cv [cm]")
    ax1.axhline(y=args.set_point, ls="--", c="r", label="set point")
    ax1.grid(True, alpha=0.3)

    (line_mv,) = ax2.plot(time_list, mv_list, c="tab:blue", label="Pump speed")
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

    fig.legend(loc="upper left", bbox_to_anchor=(0.79, 0.98))

    ax1.set_ylim(0, 8.5)
    ax2.set_ylim(0, 36)
    ax3.set_ylim(-8, 8)

    num_ticks = 5
    ticks = np.linspace(-1, 1, num_ticks)
    zero_ticks = np.linspace(0, 1, num_ticks)
    ax2.set_yticks(zero_ticks * 36)
    ax3.set_yticks(ticks * 8)

    ax2.set_xlabel("Time")
    ax2.set_xlim(
        0, args.experiment_horizon * 60 * 1.2
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
    sensor = activate_sensor(YOLO_path=r"parameters\best_new.pt", **sensor_params)

    UPPER_VALVE_PIN = config["UPPER_VALVE_PIN"]
    LOWER_VALVE_PIN = config["LOWER_VALVE_PIN"]
    PUMP_PIN = config["PUMP_PIN"]

    # For the control scenario purpose, we open both upper and lower valve for entire time
    # Initialise PyArduino
    pa = PyArduino("minima")
    # Open both valve
    pa.run_digital_write(UPPER_VALVE_PIN, True)
    pa.run_digital_write(LOWER_VALVE_PIN, True)

    # =================================================================================================
    start_time = time.perf_counter()
    prev_time = 0
    prev_cv = None
    try:
        for cv in sensor:
            now = time.perf_counter()

            proceed = delete_spike(prev_cv, cv, threshold=0.1)
            prev_cv = cv

            if proceed:
                cv_list.append(cv)
                if len(cv_list) >= 5:
                    cv = float(np.mean(cv_list[-5:]))
                    filtered_cv.append(cv)
                else:
                    filtered_cv.append(cv)
                    pass

                error = args.set_point - cv

                mv = controller_action_selection(
                    controller, args.set_point, cv, dt=now - max(start_time, prev_time)
                )

                prev_time = now
                error_list.append(error)
                time_list.append(now - start_time)
                pa.run_pump_speed(target_speed=mv)
                mv_list.append(mv)

                line_mv.set_data(time_list, mv_list)
                line_cv.set_data(time_list, cv_list)
                line_err.set_data(time_list, error_list)
                line_filter_cv.set_data(time_list, filtered_cv)

                fig.canvas.draw()
                fig.canvas.flush_events()

                if time.perf_counter() - start_time >= args.experiment_horizon * 60:
                    break

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
        "cv": cv_list,
        "filtered_cv": filtered_cv,
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

    os.makedirs(r"data\control_result", exist_ok=True)
    for i in range(2, 1000):
        filename = rf"data\control_result\{config["controller_type"]}_{i:04d}.pickle"
        if not os.path.exists(filename):
            break

    with open(filename, "wb") as f:
        pickle.dump(result_data, f)

    print("Control test result saved as:", filename)

    time.sleep(2 * 60)
    pa.run_digital_write(UPPER_VALVE_PIN, False)
    pa.run_digital_write(LOWER_VALVE_PIN, False)
