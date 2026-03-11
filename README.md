# Drone RL

Autonomous drone navigation using reinforcement learning.
Trained entirely in simulation, targeting real-world deployment.

## Current Status

| Phase | Status | Result |
|---|---|---|
| KIN navigation (MuJoCo) | ✅ Complete | **9/9 benchmark** — all cases including high_target |
| Vision navigation (MuJoCo) | 🔄 Training | CNN policy, ~100 fps, ETA ~6 hours |
| Person tracking | 📋 Next | Moving target + visual lock-on |
| Sim-to-real transfer | 📋 Planned | Domain randomization → hardware |

## 9-Case Benchmark — `mujoco_nav_9_9` (5 runs each)

| Case | Result | Dist | Steps |
|---|---|---|---|
| approach +X | ✓ 5/5 | 0.091m | 105 |
| approach −X | ✓ 5/5 | 0.093m | 193 |
| approach +Y | ✓ 5/5 | 0.096m | 260 |
| approach −Y | ✓ 5/5 | 0.085m | 115 |
| low target | ✓ 5/5 | 0.093m | 107 |
| **high target** | ✓ 5/5 | **0.075m** | **49** |
| near 0.3m | ✓ 5/5 | 0.099m | 110 |
| far 1.8m | ✓ 5/5 | 0.085m | 138 |
| diagonal | ✓ 5/5 | 0.090m | 159 |

`high_target` was PyBullet's persistent failure. Direct motor control in MuJoCo solves it.

## Architecture

- **Simulator**: MuJoCo 3.5 (8,633 fps single env — 216× faster than PyBullet)
- **Algorithm**: SAC (Soft Actor-Critic) with curriculum learning
- **KIN obs**: 16-dim state (pos, vel, quat, angvel, rel_target)
- **Vision obs**: SmallCNN + VecFrameStack n=4 → (12, 32, 48) drone camera
- **Training**: SubprocVecEnv×8 ~550 fps (KIN), DummyVecEnv×1 ~100 fps (vision)
- **Curriculum**: 0.3m → 0.6m → 1.2m → 2.0m target range

## Stack

Python 3.10 · MuJoCo 3.5 · Stable-Baselines3 2.7.1 · PyTorch · JAX/MJX (WSL2)

## Structure

```
envs/mujoco/
  hover_env.py        HoverEnv base — 16-dim obs, 4-motor direct thrust, 50Hz
  nav_env.py          NavEnv — potential reward, 3-consec goal, curriculum
  vision_nav_env.py   VisionNavEnv — (3,32,48) drone_cam obs, same reward
  quadrotor.xml       Crazyflie-inspired MJCF (75 lines, no meshes)

training/active/
  train_nav_mujoco.py       KIN nav — SAC, SubprocVecEnv×8, 2M steps ✅
  train_vision_mujoco.py    Vision nav — SmallCNN, DummyVecEnv×1, 2M steps 🔄

evaluation/
  benchmark_mujoco.py       9-case fixed benchmark
  visualize_nav_mujoco.py   MuJoCo passive viewer (live policy visualization)

models/golden/
  mujoco_nav_9_9.zip        9/9 KIN nav — current best
  mujoco_nav_8_9_best.zip   8/9 — high_target solved
  mujoco_nav_8_9_final.zip  8/9 — far_1.8m solved
```

## Quick Start

```bash
conda activate drone-rl

# Run 9-case benchmark
python evaluation/benchmark_mujoco.py

# Visualize trained policy (opens MuJoCo viewer)
python evaluation/visualize_nav_mujoco.py

# Train KIN nav from scratch (2M steps, ~2 hours)
python training/active/train_nav_mujoco.py

# Train vision nav (2M steps, ~6 hours)
python training/active/train_vision_mujoco.py
```

## Performance

| Setup | FPS |
|---|---|
| PyBullet single env | ~40 |
| MuJoCo single env | 8,633 |
| MuJoCo SubprocVecEnv×8 (KIN) | ~550 |
| MuJoCo DummyVecEnv×1 + renderer (vision) | ~216 raw / ~100 with SAC |
| MuJoCo MJX CPU vmap (256 envs) | 81,425 |
| MuJoCo MJX GPU (WSL2 + CUDA JAX) | TBD |

## Environment

- Python 3.10, conda env `drone-rl`
- MuJoCo 3.5.0, mujoco-mjx, stable-baselines3 2.7.1, gymnasium 1.2.3
- RTX 4070 Laptop (CUDA 12.8) — GPU used for SAC updates
