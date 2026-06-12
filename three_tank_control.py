import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import yaml

from sensor_control import activate_sensor
from model.PID import PID
from AILevelingRL.PyArduino import PyArduino
from analysis import analysis

from util.util import delete_spike

"""
mv = pump speed
cv = liquid level
sp = set point
"""

with open(r"parameters\steady_state_param.pickle", "rb") as f:
    steady_param = pickle.load(f)


def control_pipeline(
    sp: float = 4.0,
    pa: object = None,
    control_horizon: float = 2.0,
    threshold: float = 0.1,
    **config,
):
    controller_map = {
        "PID": PID,
    }

    if pa is None:
        raise ValueError("pa required")

    controller_mode = config.get("controller_mode")
    if controller_mode not in controller_map:
        raise ValueError(f"Unsupported controller: {controller_mode}")

    controller_params = config.get("controller")
    if controller_params is None:
        raise ValueError("Missing 'controller' configuration in config")
    controller = controller_map[controller_mode](**controller_params)

    sensor_params = config.get("sensor")
    if sensor_params is None:
        raise ValueError("Missing 'sensor' configuration in config")
    sensor = activate_sensor(YOLO_path=r"parameters\best_new.pt", **sensor_params)

    steady_state_pump_speed = np.sqrt((sp - steady_param["B"]) / steady_param["A"])

    cv_list = []
    error_list = []
    mv_list = []
    time_list = []
    filtered_cv = []

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig.subplots_adjust(right=0.78)
    (line_cv,) = ax1.plot(time_list, cv_list, label="liquid livel")
    (line_filter_cv,) = ax1.plot(time_list, filtered_cv, label="moving average")
    ax1.set_ylabel("Liquid level, cv [cm]")
    ax1.axhline(y=sp, ls="--", c="r", label="set point")
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
        0, control_horizon * 60 * 1.2
    )  # Although we have set the inference to occur every some period of time, due to the latency within the process, it usually may take longer than expected
    ax2.grid(True, alpha=0.3)

    plt.ion()
    plt.show(block=False)

    start_time = time.perf_counter()
    prev_time = 0
    prev_cv = None
    try:
        for cv in sensor:
            now = time.perf_counter()

            proceed = delete_spike(prev_cv, cv, threshold)
            prev_cv = cv

            if proceed:
                cv_list.append(cv)
                if len(cv_list) >= 5:
                    cv = float(np.mean(cv_list[-5:]))
                    filtered_cv.append(cv)
                else:
                    filtered_cv.append(cv)
                    pass

                error = sp - cv

                mv = controller.final_control(
                    sp=sp, cv=cv, dt=now - max(start_time, prev_time)
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

                if time.perf_counter() - start_time >= control_horizon * 60:
                    break

    finally:
        sensor.close()
        plt.close(fig)

    return mv_list, error_list, cv_list, time_list, filtered_cv


if __name__ == "__main__":

    control_horizon = 15.0
    set_point = 4.25

    with open(r"config\P.yaml") as f:
        config = yaml.safe_load(f)

    UPPER_VALVE_PIN = config["UPPER_VALVE_PIN"]
    LOWER_VALVE_PIN = config["LOWER_VALVE_PIN"]
    PUMP_PIN = config["PUMP_PIN"]

    # For the control scenario purpose, we open both upper and lower valve for entire time
    # Initialise PyArduino
    pa = PyArduino("minima")
    # Open both valve
    pa.run_digital_write(UPPER_VALVE_PIN, True)
    pa.run_digital_write(LOWER_VALVE_PIN, True)

    mv, err, cv, time_, filtered_cv = control_pipeline(
        sp=set_point, pa=pa, control_horizon=control_horizon, **config
    )

    pa.run_pump_speed(target_speed=0)

    ie, iae, ise, itae = analysis(err, time_)

    result_data = {
        "sp": set_point,
        "mv": mv,
        "err": err,
        "cv": cv,
        "filtered_cv": filtered_cv,
        "time": time_,
        "ie_list": np.insert(ie, 0, 0),
        "ie_final": ie[-1],
        "iae_list": np.insert(iae, 0, 0),
        "iae_final": iae[-1],
        "ise_list": np.insert(ise, 0, 0),
        "ise_final": ise[-1],
        "itae_list": np.insert(itae, 0, 0),
        "itae_final": itae[-1],
    }

    os.makedirs("control_result", exist_ok=True)

    for i in range(1, 1000):
        filename = f"control_result/{config["controller_type"]}_{i:04d}.pickle"
        if not os.path.exists(filename):
            break

    with open(filename, "wb") as f:
        pickle.dump(result_data, f)

    print("Control test result saved as:", filename)

    time.sleep(2 * 60)
    pa.run_digital_write(UPPER_VALVE_PIN, False)
    pa.run_digital_write(LOWER_VALVE_PIN, False)
