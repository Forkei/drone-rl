"""
MuJoCo Vision Nav — SAC + SmallCNN + curriculum.

VisionNavEnv (3,32,48) → VecFrameStack n=4 → (12,32,48)
DummyVecEnv×1 (OpenGL context constraint — no SubprocVecEnv).
SmallCNN features extractor (NatureCNN fails on H=32 < 36).

Curriculum: 0.3m → 0.6m → 1.2m → 2.0m (same as KIN nav)
Thresholds:  4.0 →  3.0 →  2.0 → 1.5  (lower — vision harder)
2M steps. Replay buffer saved every 200k. Frames saved every 100k.

Expected: 80-120 fps. ETA ~5-7 hours.
"""

import os, sys, time
import numpy as np

sys.path.insert(0, r"C:\Users\forke\Documents\Drones\PyBullet1")
os.chdir(r"C:\Users\forke\Documents\Drones\PyBullet1")

import torch as th
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from envs.mujoco.vision_nav_env import VisionNavEnv

# ── config ─────────────────────────────────────────────────────────────────────
STAGES      = [0.3, 0.6, 1.2, 2.0]
THRESHOLDS  = [4.0, 3.0, 2.0, 1.5]   # lower than KIN — vision is harder

SAVE_DIR    = "./models_vision_mujoco"
LOG_DIR     = "./logs_vision_mujoco"
FRAME_DIR   = "./benchmark_results/vision_frames"
TOTAL_STEPS = 2_000_000
IMG_H, IMG_W, N_STACK = 32, 48, 4


# ── SmallCNN ───────────────────────────────────────────────────────────────────

class SmallCNN(BaseFeaturesExtractor):
    """
    3-layer CNN for (C*n_stack, H, W) with H=32, W=48.
    Avoids NatureCNN's k=8s4 first layer that kills small spatial dims.

    Input:  (12, 32, 48)  [4 stacked RGB frames, channel-first]
    Output: 512-dim feature vector
    """

    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]   # 12

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input, 32, kernel_size=4, stride=2),   # → (32, 15, 23)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),         # → (64, 7, 11)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),         # → (64, 5,  9)
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.zeros(1, *observation_space.shape)
            n_flat = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flat, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(obs.float() / 255.0))


# ── callbacks ──────────────────────────────────────────────────────────────────

class ReplayBufferSaveCallback(BaseCallback):
    def __init__(self, save_dir, save_freq=200_000, verbose=1):
        super().__init__(verbose)
        self.save_dir  = save_dir
        self.save_freq = save_freq
        self._last     = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.save_freq:
            path = os.path.join(self.save_dir, "replay_buffer")
            self.model.save_replay_buffer(path)
            if self.verbose:
                print(f"  [RB] Saved @ {self.num_timesteps:,} → {path}.pkl")
            self._last = self.num_timesteps
        return True


class RedFractionLogCallback(BaseCallback):
    """Log mean red_fraction per episode to TensorBoard.
    Increasing red_fraction over training = visual attention developing.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._ep_red_fractions = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "red_fraction" in info:
                self._ep_red_fractions.append(info["red_fraction"])
            # Log at episode end
            if info.get("episode") is not None and self._ep_red_fractions:
                mean_rf = float(np.mean(self._ep_red_fractions))
                self.logger.record("vision/red_fraction_mean", mean_rf)
                self._ep_red_fractions = []
        return True


class FrameSaveCallback(BaseCallback):
    """Save a sample camera frame every save_freq steps."""

    def __init__(self, frame_dir, save_freq=100_000, verbose=1):
        super().__init__(verbose)
        self.frame_dir = frame_dir
        self.save_freq = save_freq
        self._last     = 0
        os.makedirs(frame_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.save_freq:
            try:
                from PIL import Image
                # Grab raw obs from the underlying (non-stacked) env
                raw_env = self.training_env.envs[0].env  # DummyVecEnv → Monitor → VisionNavEnv
                obs = raw_env._get_cam_frame()            # (3, H, W) uint8
                # Channel-last for PIL
                img = Image.fromarray(np.transpose(obs, (1, 2, 0)))
                path = os.path.join(self.frame_dir, f"step_{self.num_timesteps:07d}.png")
                img.save(path)
                if self.verbose:
                    print(f"  [FRAME] Saved {path}")
            except Exception as e:
                print(f"  [FRAME] Warning: {e}")
            self._last = self.num_timesteps
        return True


class EvalCurriculumCallback(EvalCallback):
    """Advances curriculum stage when eval_reward crosses threshold."""

    def __init__(self, eval_env_fn, save_dir, log_dir, **kwargs):
        eval_env = eval_env_fn()
        super().__init__(
            eval_env,
            best_model_save_path=save_dir,
            log_path=log_dir,
            eval_freq=10_000,         # eval every 10k steps (1 env is slower)
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
            **kwargs,
        )
        self.save_dir    = save_dir
        self.eval_env_fn = eval_env_fn
        self._stage      = 0
        self._t_start    = None

    def _on_training_start(self):
        super()._on_training_start()
        self._t_start = time.perf_counter()

    def _on_step(self) -> bool:
        result = super()._on_step()

        if (self.last_mean_reward != -np.inf
                and self._stage < len(STAGES) - 1
                and self.last_mean_reward >= THRESHOLDS[self._stage]):

            old_range = STAGES[self._stage]
            self._stage += 1
            new_range  = STAGES[self._stage]

            print(f"\n{'='*60}")
            print(f"CURRICULUM ADVANCE: stage {self._stage-1}→{self._stage}")
            print(f"  target_range: {old_range}m → {new_range}m")
            print(f"  eval_reward:  {self.last_mean_reward:.3f}")
            print(f"  timesteps:    {self.num_timesteps:,}")
            elapsed = time.perf_counter() - self._t_start
            print(f"  wall time:    {elapsed/60:.1f} min")
            print(f"{'='*60}\n")

            # Update train env target_range
            self.training_env.env_method("set_target_range", new_range)
            # Rebuild eval env with new range
            self.eval_env.close()
            self.eval_env = self.eval_env_fn(target_range=new_range)

            path = os.path.join(self.save_dir, f"vision_stage{self._stage}_{new_range}m")
            self.model.save(path)
            self.model.save_replay_buffer(
                os.path.join(self.save_dir, f"replay_buffer_stage{self._stage}")
            )
            print(f"  Stage checkpoint → {path}.zip")

        return result


# ── env factories ──────────────────────────────────────────────────────────────

def make_train_env(target_range=STAGES[0]):
    env = Monitor(VisionNavEnv(target_range=target_range))
    return env


def make_eval_env(target_range=STAGES[0]):
    env = VecFrameStack(
        DummyVecEnv([lambda: Monitor(VisionNavEnv(target_range=target_range))]),
        n_stack=N_STACK,
    )
    return env


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SAVE_DIR,   exist_ok=True)
    os.makedirs(LOG_DIR,    exist_ok=True)
    os.makedirs(FRAME_DIR,  exist_ok=True)

    print("[INFO] MuJoCo Vision Nav — SAC + SmallCNN + curriculum")
    print(f"[INFO] stages: {STAGES}  thresholds: {THRESHOLDS}")
    print(f"[INFO] obs: ({3*N_STACK}, {IMG_H}, {IMG_W}) → SmallCNN → 512-dim")

    train_env = VecFrameStack(
        DummyVecEnv([make_train_env]),
        n_stack=N_STACK,
    )

    policy_kwargs = dict(
        features_extractor_class=SmallCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[256, 256],
    )

    model = SAC(
        "CnnPolicy",
        train_env,
        learning_rate=1e-4,
        buffer_size=100_000,
        batch_size=64,
        learning_starts=2_000,
        train_freq=4,
        gradient_steps=1,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="auto",
    )

    print(f"[INFO] Model parameters: {sum(p.numel() for p in model.policy.parameters()):,}")

    callbacks = CallbackList([
        ReplayBufferSaveCallback(save_dir=SAVE_DIR, save_freq=200_000),
        FrameSaveCallback(frame_dir=FRAME_DIR, save_freq=100_000),
        RedFractionLogCallback(),
        EvalCurriculumCallback(
            eval_env_fn=make_eval_env,
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

    model.save(os.path.join(SAVE_DIR, "vision_final"))
    model.save_replay_buffer(os.path.join(SAVE_DIR, "replay_buffer"))
    print(f"\n[DONE] {elapsed/60:.1f} min total")
    print(f"  Final model → {SAVE_DIR}/vision_final.zip")

    train_env.close()


if __name__ == "__main__":
    main()
