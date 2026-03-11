"""
MuJoCo NavEnv — SAC + curriculum training.

Curriculum: 0.3m → 0.6m → 1.2m → 2.0m
Advances when eval_reward crosses stage threshold.
SubprocVecEnv × 8, 2M steps.
Replay buffer saved every 200k steps (mandatory).

Expected: full curriculum in ~1-2 hours.
"""

import os
import time
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs.mujoco.nav_env import NavEnv

# ── curriculum ────────────────────────────────────────────────────────────────
STAGES      = [0.3, 0.6, 1.2, 2.0]
# Advance when mean eval reward exceeds threshold for that stage.
# With goal_bonus=10, a consistent success gives ~10 per ep.
# Threshold scales down as range grows (harder).
THRESHOLDS  = [6.0, 5.0, 4.0, 3.0]

SAVE_DIR    = "./models_nav_mujoco"
LOG_DIR     = "./logs_nav_mujoco"
N_ENVS      = 8
TOTAL_STEPS = 2_000_000


def make_env(rank: int, target_range: float = 0.3):
    def _init():
        return Monitor(NavEnv(target_range=target_range))
    return _init


# ── callbacks ─────────────────────────────────────────────────────────────────

class ReplayBufferSaveCallback(BaseCallback):
    """Save replay buffer every save_freq steps — MANDATORY."""
    def __init__(self, save_dir: str, save_freq: int = 200_000, verbose=1):
        super().__init__(verbose)
        self.save_dir  = save_dir
        self.save_freq = save_freq
        self._last     = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.save_freq:
            path = os.path.join(self.save_dir, "replay_buffer")
            self.model.save_replay_buffer(path)
            if self.verbose:
                print(f"  [RB] Replay buffer saved @ {self.num_timesteps:,} → {path}.pkl")
            self._last = self.num_timesteps
        return True


class EvalCurriculumCallback(EvalCallback):
    """EvalCallback that advances curriculum stage on threshold crossing."""

    def __init__(self, train_env, eval_env, save_dir, log_dir, **kwargs):
        super().__init__(
            eval_env,
            best_model_save_path=save_dir,
            log_path=log_dir,
            eval_freq=max(20_000 // N_ENVS, 1),
            n_eval_episodes=20,
            deterministic=True,
            verbose=1,
            **kwargs,
        )
        self.train_env   = train_env
        self._stage      = 0
        self._t_start    = None

    def _on_training_start(self):
        super()._on_training_start()
        self._t_start = time.perf_counter()

    def _on_step(self) -> bool:
        result = super()._on_step()

        # Check for stage advance after each eval
        if (self.last_mean_reward != -np.inf
                and self._stage < len(STAGES) - 1
                and self.last_mean_reward >= THRESHOLDS[self._stage]):

            old_range = STAGES[self._stage]
            self._stage += 1
            new_range = STAGES[self._stage]

            print(f"\n{'='*60}")
            print(f"CURRICULUM ADVANCE: stage {self._stage-1}→{self._stage}")
            print(f"  target_range: {old_range}m → {new_range}m")
            print(f"  eval_reward:  {self.last_mean_reward:.3f} >= {THRESHOLDS[self._stage-1]}")
            print(f"  timesteps:    {self.num_timesteps:,}")
            elapsed = time.perf_counter() - self._t_start
            print(f"  wall time:    {elapsed/60:.1f} min")
            print(f"{'='*60}\n")

            # Update all train envs + eval env
            self.train_env.set_attr("target_range", new_range)
            self.eval_env.set_attr("target_range", new_range)

            # Save a stage checkpoint
            path = os.path.join(self.save_dir, f"nav_stage{self._stage}_{new_range}m")
            self.model.save(path)
            self.model.save_replay_buffer(
                os.path.join(self.save_dir, f"replay_buffer_stage{self._stage}")
            )
            print(f"  Stage checkpoint → {path}.zip")

        return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,  exist_ok=True)

    print(f"[INFO] MuJoCo NavEnv — SAC curriculum")
    print(f"[INFO] stages: {STAGES}  thresholds: {THRESHOLDS}")
    print(f"[INFO] {N_ENVS} envs × {TOTAL_STEPS:,} steps")

    train_env = SubprocVecEnv([make_env(i, STAGES[0]) for i in range(N_ENVS)])
    eval_env  = SubprocVecEnv([make_env(i, STAGES[0]) for i in range(4)])

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

    callbacks = CallbackList([
        ReplayBufferSaveCallback(save_dir=SAVE_DIR, save_freq=200_000, verbose=1),
        EvalCurriculumCallback(
            train_env=train_env,
            eval_env=eval_env,
            save_dir=SAVE_DIR,
            log_dir=LOG_DIR,
        ),
    ])

    t0 = time.perf_counter()
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=True,
    )
    elapsed = time.perf_counter() - t0

    # Final save
    model.save(os.path.join(SAVE_DIR, "nav_final"))
    model.save_replay_buffer(os.path.join(SAVE_DIR, "replay_buffer"))
    print(f"\n[DONE] {elapsed/60:.1f} min total")
    print(f"  Final model → {SAVE_DIR}/nav_final.zip")
    print(f"  Replay buffer → {SAVE_DIR}/replay_buffer.pkl")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
