# shequant_os_v1: Multimodal Latent Space Fusion for GPS-Denied Aerospace Control

**Status:** Stable / Research-Grade  
**Author:** Xander Sheppard, Principal AI Researcher  
**Organization:** Shequant  

---

## Abstract

Modern autonomous flight relies heavily on external coordinate systems (GPS) and disjointed processing pipelines. In GPS-denied environments—such as electronic warfare zones, subterranean structures, or deep-space orbital operations—standard off-the-shelf wrappers collapse because they abstract underlying aerodynamic physics away from the control policy.

`shequant_os_v1` is a bare-metal, mathematically rigorous autonomy stack designed to solve GPS-denied stabilization and trajectory control. It couples a custom C++20 aerodynamics kernel directly to a PyTorch Vision Transformer (ViT) and Proximal Policy Optimization (PPO) Actor-Critic network. Rather than relying on external position coordinates, the agent learns to stabilize and maneuver the craft by fusing raw `64x64` visual pixel arrays and internal kinematic IMU states directly inside a shared latent space.

---

## Repository Structure

```text
shequant_os_v1/
├── layer1_pilot/
│   ├── atmosphere_models/          # Trained ViT-PPO checkpoints (e.g., sheyo_ppo_150.pth)
│   ├── atmosphere_neural/policy/   # ViT encoder and PPO Actor-Critic network definitions
│   ├── atmosphere_sim/             # Native Gymnasium environment interface (sheyo_gym_atmosphere.py)
│   └── train_atmosphere.py         # Curriculum training entrypoint
├── shequant_env_standalone/
│   └── atmosphere_kernel/          # C++20 physics kernel (CMakeLists.txt, core_math.hpp, atmosphere_export.cpp)
├── mission_logs/                   # Flight telemetry logs
├── validation_logs/                # Classical and neural benchmark outputs
├── phase_validation_logs.txt       # Reproducible multi-phase validation summary
└── README.md
```

---

## Core Architecture

### 1. Bare-Metal Physics Kernel (`C++20`)

* **Location:** `shequant_env_standalone/atmosphere_kernel/`
* **Function:** Bypasses Python runtime overhead to compute nonlinear quadratic aerodynamic drag, thrust vectoring, gravity, and stochastic wind disturbances with microsecond step latency.
* **Curriculum Dynamics:** Supports real-time mid-flight environmental shifts (e.g., transitioning from Earth gravity $-9.81\text{ m/s}^2$ to hyper-gravity $-20.0\text{ m/s}^2$) alongside randomized vehicle mass ($0.8\text{–}1.2\text{ kg}$), actuator efficiency ($0.8\text{–}1.2$), and drag coefficients ($C_d \in [0.10, 0.15]$).

### 2. Multimodal Neural Policy (`PyTorch`)

* **Location:** `layer1_pilot/atmosphere_neural/policy/`
* **Vision Transformer (ViT):** Patch-embeds and encodes `64x64` RGB environmental frames into spatial-contextual tokens.
* **Kinematics & PPO Actor-Critic:** Encodes 4D internal inertial/state telemetry ($[v_z, \omega_x, \omega_y, \Delta z]$) and fuses it with the ViT latent representation to output continuous 3D spatial thrust commands.

---

## Validation & Ablation Benchmarks

Extensive Monte Carlo and disturbance-rejection benchmarks confirm both the fidelity of the C++ plant and the necessity of multimodal visual-inertial fusion:

| Controller / Policy Mode | Plant Conditions | Settling Rate (<10 s) | Median Settling Time | Median Steady-State Error | Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **P Baseline** ($K_p = 2.5$, no FF) | 300 Randomized Runs | 20% | $>10.00\text{ s}$ | $0.208\text{ m}$ | Gravity sag without feedforward |
| **PD+FF Baseline** ($K_p = 2.5, K_d = 0.5$) | 300 Randomized Runs | 87% | $0.57\text{ s}$ | $0.023\text{ m}$ | Sensitive to mass/efficiency mismatch |
| **PID+FF Baseline** ($K_i = 1.0$, anti-windup) | 300 Randomized Runs | **100%** | $1.00\text{ s}$ | $0.002\text{ m}$ | Mathematical control ceiling ($p_{90} = 0.003\text{ m}$) |
| **ViT-PPO (Blind Kinematic Ablation)** | Nominal Quadratic Plant | 0% | $>10.00\text{ s}$ | $59.238\text{ m}$ | Zeroed vision tensor saturates thrust ($68.95\text{ m}$ overshoot) |
| **ViT-PPO (Full Multimodal Fusion)** | Live RGB + IMU Stream | — | — | — | *Full quantitative results detailed in upcoming publication* |

---

## Build & Reproducibility

This repository enforces deterministic seed locking, relative path resolution, and clean separation between the C++ physics shared library and PyTorch training loop.

### 1. Build the C++ Physics Kernel

```bash
cmake -S shequant_env_standalone/atmosphere_kernel -B shequant_env_standalone/build -DCMAKE_BUILD_TYPE=Release
cmake --build shequant_env_standalone/build --config Release
```

### 2. Install Python Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy torch gymnasium
```

### 3. Run Curriculum Training

```bash
python3 layer1_pilot/train_atmosphere.py
```

---

## Citation

If you reference `shequant_os_v1` in academic or technical work, please cite:

```bibtex
@software{sheppard2026shequant,
  author       = {Sheppard, Xander},
  title        = {shequant\_os\_v1: Multimodal Latent Space Fusion for GPS-Denied Aerospace Control},
  year         = {2026},
  publisher    = {Shequant},
  version      = {1.0.0}
}
```