# Three-Tank Liquid-Level Control

## Overview

This repository documents a research-practice project on liquid-level regulation using a three-tank experimental rig. It compares conventional feedback controllers, model predictive control (MPC), and reinforcement-learning (RL) controllers under nominal and non-nominal operating conditions.

The project covers the full workflow from experimental data collection and system identification to controller tuning/training, hardware-facing experiments, and result analysis.

### Controllers evaluated

- **Classical control:** P, PI, and PID
- **Model-based control:** MPC with Kalman-filter state estimation
- **Reinforcement learning:** DQN, recurrent PPO, and SAC

### Evaluation scenarios

- Nominal reference tracking
- Abnormal initial conditions
- Disturbances applied to the upper tank
- Disturbances applied to the lower tank
- Disturbance-free tracking at a changed reference level

Performance is examined using tracking behavior, integral error metrics such as **IAE** and **ISE**, and actuator-effort indicators.

> **Results and interpretation:** For the complete experimental figures, quantitative comparisons, and discussion, open [`result_presentation.ipynb`](./result_presentation.ipynb).

---

## Repository Structure

```text
.
├── result_presentation.ipynb       # Main results, figures, metrics, and interpretation
├── control.py                      # Nominal-condition control experiment
├── control_abnormal_init.py        # Control experiment with abnormal initial conditions
├── control_w_disturbance.py        # Control experiment with upper/lower-tank disturbances
├── disturbance_only.py             # Disturbance-characterization experiment
├── sensor_control.py               # Sensor-interface utilities
├── analysis.py                     # Data processing and performance analysis utilities
├── config/                         # Controller hyperparameters and tuning configurations
│   ├── P.yaml
│   ├── PI.yaml
│   ├── PID.yaml
│   ├── MPC.yaml
│   ├── DQN.yaml
│   ├── recurrentPPO.yaml
│   └── SAC.yaml
├── models/                         # Controller and process-model implementations
│   ├── PID.py
│   ├── MPC.py
│   ├── RL.py
│   ├── recurrentPPO.py
│   ├── SAC.py
│   └── experiment_model.py
├── parameters/                     # Identified model parameters and trained model weights
│   ├── state_space_param.pickle
│   ├── steady_state_param.pickle
│   ├── DQN.pt
│   ├── recurrentPPO.pt
│   └── SAC.pt
├── train/                          # Training and tuning notebooks
│   ├── StateSpaceModel_train.ipynb
│   ├── PID_tuning.ipynb
│   ├── MPC_tuning.ipynb
│   ├── DQN_train.ipynb
│   ├── PPO_train.ipynb
│   ├── recurrentPPO_train.ipynb
│   └── SAC_train.ipynb
├── data/                           # Experimental data, saved control runs, and setup images
│   ├── constant_pump_speed_experiment/
│   ├── control_result/
│   ├── setup_image/
│   └── util/
├── util/                           # Supporting identification and inference notebooks/utilities
│   ├── steady_state_map/
│   └── YOLO_inference_linear_reg/
├── AILevelingRL/                   # Arduino, camera/vision, and hardware-support code
├── requirements.txt                # Python dependencies
└── environment.yaml                # Conda environment definition
```

---

## Getting Started

### 1. Create the environment

This project was developed with **Python 3.12**.

```bash
conda env create -f environment.yaml
conda activate three_tank_control
```

Alternatively, create a virtual environment and install the dependencies manually:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

### 2. Review the results

Launch Jupyter and open the main presentation notebook:

```bash
jupyter lab
```

Then open:

```text
result_presentation.ipynb
```

The saved experimental results in `data/control_result/` allow the result notebook to be reviewed without connecting to the physical apparatus.

### 3. Reproduce or extend experiments

The primary experiment scripts are:

```bash
python control.py --controller PID --initial_state 1.5 1.5 --set_point 4.5
python control_abnormal_init.py --controller SAC --initial_state 1.5 1.5 --set_point 4.5
python control_w_disturbance.py --controller MPC --initial_state 1.5 1.5 --set_point 6.0
```

Supported controller options are:

```text
P, PI, PID, DQN, RPPO, SAC, MPC
```

> **Hardware note:** These scripts are coupled to a lab-specific experimental setup. Before running a live experiment, review the sensor interface, serial/Arduino configuration, pump limits, safety interlocks, and controller configuration files. The code was developed in a Windows environment; some scripts use Windows-style paths and may require path adjustments on other operating systems.

---

## Workflow

1. **Collect experimental data** using constant pump-speed trials.
2. **Identify steady-state and dynamic models** using the notebooks in `util/` and `train/`.
3. **Tune classical and MPC controllers** using the corresponding notebooks and YAML configurations.
4. **Train RL controllers** using the DQN, PPO, recurrent PPO, and SAC training notebooks.
5. **Run control experiments** under nominal, abnormal-initial-state, and disturbance scenarios.
6. **Analyze outcomes** in [`result_presentation.ipynb`](./result_presentation.ipynb).

---

## Scope and Limitations

- The reported results are specific to the experimental rig, controller implementations, reward designs, tuning choices, and disturbance scenarios in this repository.
- RL policies were trained under particular operating assumptions; performance can degrade under distribution shift, unmodeled disturbances, or reference changes.
- This repository is intended for research and educational use. It is not a safety-certified control system and should not be deployed on an industrial process without independent validation, safeguards, and supervision.

---

## Acknowledgment

I sincerely thank Prof. Taehoon Oh for the valuable opportunity to undertake this research practice. His guidance enabled me to gain hands-on experience with various system-identification methods and control algorithms.
