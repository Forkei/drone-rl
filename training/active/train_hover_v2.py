"""
MuJoCo HoverEnv v2 — SubprocVecEnv×8, hover-centered action, angvel penalty.

Changes vs v1:
  - SubprocVecEnv × 8 parallel envs
  - action=0 → hover thrust (easier starting point)
  - reward -= 0.1 * ||angvel|| (anti-tumble)
  - 500k steps with eval at 100k / 300k / 500k

Expected: ~400-600 fps, stable hover by 200k steps.
"""

import os
import time
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from envs.mujoco.hover_env import HoverEnv

SAVE_DIR = "./models_hover_v2"
LOG_DIR  = "./logs_hover_v2"
N_ENVS   = 8
TOTAL_STEPS = 500_000
MILESTONES  = [100_000, 300_000, 500_000]


def make_env(rank: int):
    def _init():
        env = Monitor(HoverEnv(target_range=0.5))  # start close
        return env
    return _init


class MilestoneCallback(BaseCallback):
    """Prints detailed stats at milestone timesteps."""
    def __init__(self, milestones, eval_env, verbose=1):
        super().__init__(verbose)
        self.milestones  = sorted(milestones)
        self.eval_env    = eval_env
        self._next_idx   = 0
        self._t_start    = None

    def _on_training_start(self):
        self._t_start = time.perf_counter()

    def _on_step(self) -> bool:
        if self._next_idx >= len(self.milestones):
            return True
        if self.num_timesteps < self.milestones[self._next_idx]:
            return True

        step = self.milestones[self._next_idx]
        self._next_idx += 1
        elapsed = time.perf_counter() - self._t_start
        fps = self.num_timesteps / elapsed

        # Quick 20-episode eval
        successes, ep_lens, dists = [], [], []
        obs = self.eval_env.reset()
        for _ in range(20):
            done = False
            ep_r, ep_len = 0, 0
            o, _ = self.eval_env.reset()
            while not done:
                a, _ = self.model.predict(o, deterministic=True)
                o, r, term, trunc, info = self.eval_env.step(a)
                ep_r += r; ep_len += 1
                done = term or trunc
            successes.append(info["success"])
            ep_lens.append(ep_len)
            dists.append(info["dist_to_target"])

        print(f"\n{'='*55}")
        print(f"MILESTONE: {step:,} steps")
        print(f"  wall time:       {elapsed/60:.1f} min")
        print(f"  fps:             {fps:.0f} steps/sec")
        print(f"  eval ep_len:     {np.mean(ep_lens):.1f} steps  (500=no crash)")
        print(f"  eval mean_dist:  {np.mean(dists):.3f}m")
        print(f"  success rate:    {np.mean(successes)*100:.1f}%")
        print(f"{'='*55}\n")
        return True


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"[INFO] SubprocVecEnv × {N_ENVS}  |  {TOTAL_STEPS:,} steps")
    print(f"[INFO] action=0 → hover thrust  |  angvel penalty active")

    # ── envs ──────────────────────────────────────────────────────────────
    train_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    eval_env  = Monitor(HoverEnv(target_range=0.5))

    # ── quick FPS pre-check ───────────────────────────────────────────────
    print("\n[FPS pre-check] 10k random steps across 8 envs...")
    obs = train_env.reset()
    t0 = time.perf_counter()
    for _ in range(1250):   # 1250 × 8 envs = 10k steps
        obs, _, _, _ = train_env.step(
            np.array([train_env.action_space.sample() for _ in range(N_ENVS)])
        )
    pre_fps = 10_000 / (time.perf_counter() - t0)
    print(f"  {pre_fps:.0f} steps/sec  (8 envs, random policy)")
    train_env.reset()

    # ── model ─────────────────────────────────────────────────────────────
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=500_000,
        batch_size=256,
        learning_starts=10_000,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="auto",
    )

    callbacks = [
        MilestoneCallback(MILESTONES, eval_env, verbose=1),
        EvalCallback(
            Monitor(HoverEnv(target_range=0.5)),
            best_model_save_path=SAVE_DIR,
            log_path=LOG_DIR,
            eval_freq=max(50_000 // N_ENVS, 1),
            n_eval_episodes=20,
            deterministic=True,
            verbose=0,
        ),
    ]

    # ── train ─────────────────────────────────────────────────────────────
    t_start = time.perf_counter()
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=True,
    )
    t_total = time.perf_counter() - t_start
    train_fps = TOTAL_STEPS / t_total

    model.save(os.path.join(SAVE_DIR, "hover_v2_final"))
    model.save_replay_buffer(os.path.join(SAVE_DIR, "replay_buffer"))

    # ── final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("FINAL SUMMARY")
    print(f"  Total time:         {t_total/60:.1f} min")
    print(f"  Training fps:       {train_fps:.0f} steps/sec")
    print(f"  Env pre-check fps:  {pre_fps:.0f} steps/sec")
    print(f"{'='*55}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
