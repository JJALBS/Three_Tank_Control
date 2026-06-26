# Three-Tank Control: System Identification and Real-Time Control

> **Acknowledgment**
> I sincerely thank Prof. Taehoon Oh for the valuable opportunity to undertake this research practice. His guidance enabled me to gain hands-on experience with various system-identification methods and control algorithms.

## Overview

This repository contains a five-month hands-on project on real-time liquid-level control using a three-tank experimental rig. The work covers system identification, vision-based level measurement, controller implementation, hardware deployment, and experimental comparison.

The main objective is to implement and compare seven control approaches on the physical rig:

* **P, PI, and PID control**
* **Model Predictive Control (MPC)**
* **Deep Q-Network (DQN)**
* **Recurrent Proximal Policy Optimization (recurrent PPO)**
* **Soft Actor-Critic (SAC)**

Performance is evaluated from real experimental runs using tracking-error metrics, including IE, IAE, ISE, and ITAE.

## Repository Structure

```text
Three_Tank_Control/
├── config/                         # YAML configurations for each controller and sensor/hardware settings
├── models/                         # Implementations of PID, MPC/Kalman filter, DQN/PPO, SAC, and state-space models
├── train/                          # Notebooks for controller training, model fitting, and parameter tuning
├── data/
│   ├── constant_pump_speed_experiment/  # Input-output data used for system identification
│   ├── control_result/                  # Saved closed-loop results: normal experiment condition, abnormal initial condition, disturbance on hidden state
│   └── util/                            # Calibration and linear-regression data
├── parameters/                     # Identified-model parameters, trained RL agents, and YOLO weights
├── util/                           # Calibration, steady-state mapping, and signal-processing utilities
├── AILevelingRL/                   # Arduino communication, firmware, and vision-control support code
├── control.py                      # Main closed-loop control experiment
├── control_abnormal_init.py        # Closed-loop experiment with abnormal initial conditions/disturbance
├── sensor_control.py               # YOLO-based water-level measurement from camera input
├── prepare_exp.py                  # Hardware preparation utility
├── reset_exp.py                    # Hardware reset utility
├── result_presentation.ipynb       # Visualization and comparison of experimental results
├── analysis.py                     # Calculation of IE, IAE, ISE, and ITAE
├── environment.yaml                # Conda environment definition
└── requirements.txt                # Python dependencies
```

## Workflow

1. Collect input-output data with `system_identification_exp.py`.
2. Fit and tune process models and controllers using notebooks in `train/`.
3. Select a controller configuration in `config/`.
4. Run real-time experiments with `control.py` or `control_abnormal_init.py`.
5. Compare saved results using `result_presentation.ipynb`.

## Notes

* Water level is measured using a YOLO-based camera sensor, while pumps and valves are controlled through an Arduino interface.
* The main experiment script supports: `P`, `PI`, `PID`, `MPC`, `DQN`, `rPPO`, and `SAC`.
* Review camera indices, pin assignments, trained weights, and safety limits before running the hardware on another setup.
