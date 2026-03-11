# drone-rl

Quadrotor navigation with deep RL — SAC + MuJoCo.

## Status
- **Phase 1-2 (PyBullet KIN)**: Complete. SAC 8/9 cases. Golden model: `models/golden/kin_nav_8_9.zip`.
- **Phase 3 (PyBullet Vision)**: Complete. SAC_3 2M steps, 1.5% success. Model: `models/golden/vision_nav_sac3.zip`.
- **Phase 4 (MuJoCo)**: Active. 8,633 fps env throughput (216× PyBullet). Training in progress.

## Structure

```
envs/
  pybullet/     PyBullet environments (archived)
  mujoco/       Active MuJoCo environments
    hover_env.py      HoverEnv — 16-dim obs, 4-motor direct thrust, 50Hz
    quadrotor.xml     Crazyflie-inspired MJCF (75 lines, no meshes)

training/
  active/       Current training scripts
  archive/      PyBullet-era scripts (reference)

evaluation/     Benchmarks, visualization
utils/          Diagnostics, monitoring
docs/           Project notes
models/golden/  Best checkpoints (gitignored)
```

## Quick Start

```bash
conda activate drone-rl
python training/active/train_hover_v2.py   # 500k steps, 8 envs
```

## Environment

- Python 3.10, conda env `drone-rl`
- MuJoCo 3.5.0 + mujoco-mjx 3.5.0 + JAX 0.6.2
- stable-baselines3 2.7.1, gymnasium 1.2.3
- RTX 4070 Laptop GPU (CUDA 12.8)

## Performance

| Setup | FPS |
|---|---|
| PyBullet single env | ~40 |
| MuJoCo single env (random) | 8,633 |
| MuJoCo SAC training (1 env) | ~54 |
| MuJoCo SubprocVecEnv×8 | ~500 (target) |
| MuJoCo MJX GPU | 50k–500k (Linux/WSL2 needed for CUDA JAX) |
