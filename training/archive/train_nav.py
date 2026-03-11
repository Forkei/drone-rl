"""
Navigation Training with Curriculum.

Stages:
  1. 0.3m radius  → advance when success_rate >= 0.60 over last 100 episodes
  2. 0.6m radius  → advance when success_rate >= 0.60
  3. 1.2m radius  → advance when success_rate >= 0.60
  4. 2.0m radius  → final stage

Usage:
    python train_nav.py [--timesteps 2000000]
"""

import argparse
import os
import numpy as np
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)


class EvalCurriculumCallback(EvalCallback):
    """EvalCallback that forwards eval mean reward to NavCallback for curriculum gating."""

    def __init__(self, nav_cb: "NavCallback", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nav_cb = nav_cb

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.last_mean_reward != -np.inf:
            self.nav_cb.notify_eval_reward(self.last_mean_reward)
        return result
from stable_baselines3.common.monitor import Monitor

from envs.nav_aviary import NavAviary


# ── Curriculum stages ─────────────────────────────────────────────────────────

CURRICULUM_STAGES = [
    # success_threshold: training rollout success rate (stochastic policy — noisy)
    # eval_reward_threshold: mean eval reward (deterministic policy — reliable signal)
    # Stage advances when EITHER threshold is met.
    {"target_range": 0.3, "success_threshold": 0.20, "eval_reward_threshold": 3.0,  "label": "stage1_0.3m"},
    {"target_range": 0.6, "success_threshold": 0.35, "eval_reward_threshold": 4.0,  "label": "stage2_0.6m"},
    {"target_range": 1.2, "success_threshold": 0.50, "eval_reward_threshold": 5.0,  "label": "stage3_1.2m"},
    {"target_range": 2.0, "success_threshold": 0.60, "eval_reward_threshold": 6.0,  "label": "stage4_2.0m"},
]


# ── Curriculum + metrics callback ─────────────────────────────────────────────

class NavCallback(BaseCallback):
    """
    - Tracks success_rate over rolling window AND last eval reward
    - Advances curriculum when EITHER training success_rate OR eval reward crosses threshold
    - This handles the gap between noisy stochastic rollouts and deterministic eval
    - Logs success_rate, mean_final_dist, curriculum_stage to TensorBoard
    """

    def __init__(self, train_envs, save_dir: str,
                 window: int = 100, check_freq: int = 2048, verbose: int = 1):
        super().__init__(verbose)
        self.train_envs    = train_envs
        self.save_dir      = save_dir
        self.window        = window
        self.check_freq    = check_freq
        self.stage_idx     = 0
        self.last_eval_reward = -np.inf   # updated by EvalCallback via shared parent

        self._successes = deque(maxlen=window)
        self._ep_dists  = deque(maxlen=window)

    @property
    def current_stage(self):
        return CURRICULUM_STAGES[self.stage_idx]

    def notify_eval_reward(self, mean_reward: float):
        """Called by EvalCurriculumCallback after each eval to share the result."""
        self.last_eval_reward = mean_reward

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._successes.append(float(info.get("success", 0.0)))
                if "dist_to_target" in info:
                    self._ep_dists.append(info["dist_to_target"])

        if self.n_calls % self.check_freq == 0 and len(self._successes) >= 10:
            success_rate = float(np.mean(self._successes))
            mean_dist    = float(np.mean(self._ep_dists)) if self._ep_dists else float("nan")

            self.logger.record("nav/success_rate",    success_rate)
            self.logger.record("nav/mean_final_dist", mean_dist)
            self.logger.record("nav/curriculum_stage", self.stage_idx)
            self.logger.record("nav/last_eval_reward", self.last_eval_reward)

            stage  = self.current_stage
            s_thr  = stage["success_threshold"]
            e_thr  = stage["eval_reward_threshold"]
            if self.verbose:
                print(f"\n[Nav] {stage['label']} | "
                      f"train_success={success_rate:.2f}/{s_thr:.2f} | "
                      f"eval_reward={self.last_eval_reward:.2f}/{e_thr:.2f} | "
                      f"mean_dist={mean_dist:.3f}m")

            if self.stage_idx < len(CURRICULUM_STAGES) - 1:
                if success_rate >= s_thr or self.last_eval_reward >= e_thr:
                    trigger = ("train_success" if success_rate >= s_thr
                               else "eval_reward")
                    self._advance_stage(trigger)

        return True

    def _advance_stage(self, trigger: str = ""):
        label = self.current_stage["label"]
        ckpt  = os.path.join(self.save_dir, f"nav_{label}.zip")
        self.model.save(ckpt)
        print(f"\n[Curriculum] {label} -> COMPLETE (trigger: {trigger}) | ckpt -> {ckpt}")

        self.stage_idx += 1
        new_range = self.current_stage["target_range"]
        self.train_envs.env_method("_set_target_range", new_range)
        print(f"[Curriculum] -> {self.current_stage['label']}  (radius {new_range}m)\n")

        self._successes.clear()
        self._ep_dists.clear()
        self.last_eval_reward = -np.inf


# ── env factory ───────────────────────────────────────────────────────────────

def make_nav_env(target_range: float = 0.3, gui: bool = False):
    def _init():
        env = NavAviary(
            target_range=target_range,
            warm_zone_bonus=0.5,    # keep drone incentivised to stay near goal
            gui=gui,
        )
        return Monitor(env)
    return _init


# ── training ──────────────────────────────────────────────────────────────────

def train(
    total_timesteps: int = 2_000_000,
    n_envs: int = 8,
    save_dir: str = "./models_nav",
    log_dir: str  = "./logs_nav",
    resume_path: str = "",
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    # Add helper for curriculum range updates
    NavAviary._set_target_range = lambda self, r: setattr(self, "target_range", r)

    initial_range = CURRICULUM_STAGES[0]["target_range"]
    train_env = make_vec_env(make_nav_env(target_range=initial_range), n_envs=n_envs)
    eval_env  = make_nav_env(target_range=initial_range)()

    if resume_path and os.path.isfile(resume_path + ".zip"):
        print(f"[INFO] Resuming from {resume_path}.zip")
        model = PPO.load(resume_path, env=train_env,
                         tensorboard_log=log_dir, verbose=1)
    else:
        model = PPO(
            "MlpPolicy", train_env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,       # conservative — prevents post-convergence collapse
            clip_range_vf=0.1,    # also clip value function
            ent_coef=0.01,        # back to 0.01 — 0.005 caused collapse in PID run
            policy_kwargs=dict(net_arch=[256, 256]),
            device="cpu",
            seed=42,
        )

    nav_cb = NavCallback(
        train_envs=train_env,
        save_dir=save_dir,
        window=100,
        check_freq=2048,
        verbose=1,
    )
    callbacks = CallbackList([
        nav_cb,
        EvalCurriculumCallback(
            nav_cb=nav_cb,
            eval_env=eval_env,
            best_model_save_path=save_dir,
            log_path=log_dir,
            eval_freq=max(20_000 // n_envs, 1),
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(50_000 // n_envs, 1),
            save_path=save_dir,
            name_prefix="ppo_nav",
        ),
    ])

    print(f"\n[INFO] Nav training — {total_timesteps:,} steps, {n_envs} envs")
    print(f"[INFO] Curriculum stages: {[s['label'] for s in CURRICULUM_STAGES]}")
    print(f"[INFO] Advance: train success >= threshold OR eval reward >= threshold (whichever first)")
    print(f"[INFO] Reward: potential-based + warm_zone +10 bonus on goal")
    print(f"[INFO] clip_range=0.1, ent_coef=0.01, tilt_limit=0.8rad\n")

    model.learn(total_timesteps=total_timesteps,
                callback=callbacks, progress_bar=True)

    final = os.path.join(save_dir, "ppo_nav_final")
    model.save(final)
    print(f"\n[INFO] Final model -> {final}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--n-envs",    type=int, default=8)
    p.add_argument("--save-dir",  default="./models_nav")
    p.add_argument("--log-dir",   default="./logs_nav")
    p.add_argument("--resume",    default="",
                   help="Path to nav model checkpoint to resume (without .zip)")
    args = p.parse_args()

    train(args.timesteps, n_envs=args.n_envs,
          save_dir=args.save_dir, log_dir=args.log_dir,
          resume_path=args.resume)
