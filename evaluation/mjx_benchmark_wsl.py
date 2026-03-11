"""
MJX GPU benchmark — run inside WSL2 Ubuntu with CUDA JAX.

Usage:
  source ~/drone-jax/bin/activate
  python evaluation/mjx_benchmark_wsl.py

Reports steps/sec at 1024, 4096, 8192 envs using quadrotor.xml.
Decision threshold: >200k fps @ 1024 envs → migrate to JAX training.
"""

import os, time, sys
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np

print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# Load quadrotor model
XML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs", "mujoco", "quadrotor.xml")
if not os.path.exists(XML_PATH):
    # Fallback: minimal model
    XML = """<mujoco><option timestep="0.002" integrator="RK4"/>
    <worldbody><body name="drone" pos="0 0 1"><freejoint/>
    <inertial mass="0.027" pos="0 0 0" diaginertia="1.4e-5 1.4e-5 2.17e-5"/>
    </body></worldbody>
    <actuator>
      <general site="rotor0" gear="0 0 1 0 0 0" ctrlrange="0 0.149"/>
    </actuator></mujoco>"""
    model = mujoco.MjModel.from_xml_string(XML)
    print("WARNING: quadrotor.xml not found, using minimal model")
else:
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    print(f"Model: {XML_PATH}")

mx = mjx.put_model(model)
data = mujoco.MjData(model)
dx = mjx.put_data(model, data)

print(f"\nModel: {model.nu} actuators, {model.nq} qpos, {model.nv} qvel")
print(f"Timestep: {model.opt.timestep}s\n")

WARMUP_STEPS = 100
BENCH_STEPS  = 1000

results = {}

for N in [1024, 4096, 8192]:
    # Build batch by stacking N copies
    batch = jax.tree_util.tree_map(
        lambda x: jnp.stack([x] * N), dx
    )
    actions = jnp.zeros((N, model.nu))

    @jax.jit
    def step_batch(b):
        return jax.vmap(lambda d: mjx.step(mx, d))(b)

    # Warmup
    for _ in range(WARMUP_STEPS):
        batch = step_batch(batch)
    jax.block_until_ready(batch.qpos)

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(BENCH_STEPS):
        batch = step_batch(batch)
    jax.block_until_ready(batch.qpos)
    elapsed = time.perf_counter() - t0

    fps = N * BENCH_STEPS / elapsed
    results[N] = fps
    print(f"  {N:>5} envs: {fps:>12,.0f} fps  ({elapsed:.2f}s for {BENCH_STEPS} steps)")

print(f"\n{'='*50}")
print("DECISION:")
fps_1024 = results.get(1024, 0)
if fps_1024 > 200_000:
    print(f"  {fps_1024:,.0f} fps @ 1024 envs → MIGRATE to JAX training loop")
elif fps_1024 > 100_000:
    print(f"  {fps_1024:,.0f} fps @ 1024 envs → MARGINAL (consider for long runs)")
else:
    print(f"  {fps_1024:,.0f} fps @ 1024 envs → SubprocVecEnv×8 is sufficient")
print(f"{'='*50}")
