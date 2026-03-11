#!/bin/bash
# Run inside WSL2 Ubuntu to set up CUDA JAX + MJX
# Usage: bash setup_wsl_jax.sh
set -e

echo "[1] Update apt"
sudo apt-get update -qq

echo "[2] Install Python pip if needed"
sudo apt-get install -y python3-pip python3-venv

echo "[3] Create venv"
python3 -m venv ~/drone-jax
source ~/drone-jax/bin/activate

echo "[4] Install JAX with CUDA 12"
pip install --upgrade pip
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo "[5] Install mujoco + mujoco-mjx"
pip install mujoco mujoco-mjx

echo "[6] Test JAX GPU"
python3 -c "import jax; print('JAX devices:', jax.devices()); print('Default backend:', jax.default_backend())"

echo "[7] MJX benchmark (256 envs)"
python3 - <<'PYEOF'
import os
os.environ['MUJOCO_GL'] = 'egl'
import time, numpy as np, mujoco, mujoco.mjx as mjx, jax, jax.numpy as jnp

xml = """
<mujoco><option timestep="0.002"/><worldbody>
  <body><freejoint/><inertial mass="0.027" diaginertia="1.4e-5 1.4e-5 2.17e-5"/></body>
</worldbody></mujoco>"""
model = mujoco.MjModel.from_xml_string(xml)
mx = mjx.put_model(model)
data = mujoco.MjData(model)
dx = mjx.put_data(model, data)

N = 256
batch = jax.vmap(lambda d: d)(jax.tree_util.tree_map(lambda x: jnp.stack([x]*N), dx))
step_batch = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
_ = step_batch(mx, batch)   # warmup

t0 = time.perf_counter()
for _ in range(1000):
    batch = step_batch(mx, batch)
batch[0].qpos.block_until_ready()
elapsed = time.perf_counter() - t0
fps = N * 1000 / elapsed
print(f"MJX GPU {N} envs: {fps:,.0f} fps")
PYEOF

echo "DONE. If you see CudaDevice above, GPU JAX is working."
