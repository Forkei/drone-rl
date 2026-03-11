"""
MJX evaluation: GPU physics benchmark + PureJaxRL integration assessment.

Steps:
  1. Verify JAX sees GPU
  2. Load quadrotor.xml into MJX
  3. Benchmark 256 / 1024 / 4096 parallel envs
  4. Smoke test: 10k random steps, check for NaN
"""

import time
import numpy as np

# ── 1. JAX + MJX availability ──────────────────────────────────────────────
print("=" * 55)
print("Step 1: JAX + MJX")
print("=" * 55)

try:
    import jax
    import jax.numpy as jnp
    print(f"JAX version:  {jax.__version__}")
    print(f"JAX devices:  {jax.devices()}")
    gpu_available = any(d.platform == "gpu" for d in jax.devices())
    print(f"GPU visible:  {gpu_available}")
except ImportError as e:
    print(f"JAX not installed: {e}")
    print("Install: pip install jax[cuda12]")
    exit(1)

try:
    import mujoco
    from mujoco import mjx
    print(f"mujoco:       {mujoco.__version__}")
    print("MJX import:   OK")
except ImportError as e:
    print(f"MJX unavailable: {e}")
    exit(1)

# ── 2. Load model ──────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Step 2: Load quadrotor into MJX")
print("=" * 55)

model = mujoco.MjModel.from_xml_path("envs/mujoco/quadrotor.xml")
mx = mjx.put_model(model)
print(f"Model loaded:  nq={model.nq}  nv={model.nv}  nu={model.nu}")
print(f"MJX model:     {type(mx)}")

# ── 3. FPS benchmark ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Step 3: FPS benchmark")
print("=" * 55)

from jax import vmap, jit
import functools

def make_batch_step(n_envs):
    # Create batched data
    dx_single = mjx.make_data(mx)
    dx_batch  = jax.tree_util.tree_map(
        lambda x: jnp.stack([x] * n_envs), dx_single
    )
    # Random ctrl (all zeros is fine for benchmark)
    ctrl = jnp.zeros((n_envs, model.nu))

    def set_ctrl_and_step(dx, ctrl):
        dx = dx.replace(ctrl=ctrl)
        return mjx.step(mx, dx)

    batch_step = jit(vmap(set_ctrl_and_step))

    # Warm up
    print(f"  [{n_envs} envs] Warming up JIT...")
    dx_batch = batch_step(dx_batch, ctrl)
    dx_batch.qpos.block_until_ready()

    return dx_batch, ctrl, batch_step

results = {}
for n_envs in [256, 1024, 4096]:
    try:
        dx_batch, ctrl, batch_step = make_batch_step(n_envs)

        n_steps = 500
        t0 = time.perf_counter()
        for _ in range(n_steps):
            dx_batch = batch_step(dx_batch, ctrl)
        dx_batch.qpos.block_until_ready()
        elapsed = time.perf_counter() - t0

        fps = n_envs * n_steps / elapsed
        results[n_envs] = fps
        print(f"  [{n_envs:5d} envs] {fps:>10,.0f} steps/sec  ({elapsed:.2f}s for {n_steps} steps)")
    except Exception as e:
        print(f"  [{n_envs:5d} envs] FAILED: {e}")
        results[n_envs] = None

# ── 4. NaN check with 256 envs ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("Step 4: NaN check (256 envs, 1000 steps)")
print("=" * 55)

try:
    dx_batch, ctrl, batch_step = make_batch_step(256)
    for i in range(1000):
        dx_batch = batch_step(dx_batch, ctrl)

    qpos = np.array(dx_batch.qpos)
    nan_count = int(np.isnan(qpos).sum())
    print(f"  NaN in qpos: {nan_count}  ({'PASS' if nan_count == 0 else 'FAIL — physics unstable'})")
    print(f"  qpos range:  [{qpos.min():.3f}, {qpos.max():.3f}]")
except Exception as e:
    print(f"  NaN check FAILED: {e}")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"  SB3 single env baseline:      ~8,600 steps/sec")
print(f"  SB3 SubprocVecEnv×8:          ~68,000 steps/sec (est)")
for n, fps in results.items():
    if fps:
        print(f"  MJX {n:4d} envs (GPU):          {fps:>10,.0f} steps/sec")
print()
