"""
Resume nav curriculum from best_model.zip + replay_buffer.pkl.
Training crashed at ~1.74M steps during curriculum advance (save_dir bug — now fixed).
Resume for 400k more steps (to ~2M total equivalent) starting at stage 1 (0.6m).
"""

import os
import time
import sys
import numpy as np

sys.path.insert(0, r"C:\Users\forke\Documents\Drones\PyBullet1")
os.chdir(r"C:\Users\forke\Documents\Drones\PyBullet1")

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs.mujoco.nav_env import NavEnv

# Import callbacks from training script
exec(open("training/active/train_nav_mujoco.py").read().split("def main():")[0])

SAVE_DIR = "./models_nav_mujoco"
LOG_DIR  = "./logs_nav_mujoco"
N_ENVS   = 8
# Resume at stage 1 (0.6m) — curriculum already crossed 6.0 threshold at 1.74M
RESUME_STAGE = 1
RESUME_RANGE = STAGES[RESUME_STAGE]
RESUME_STEPS = 400_000

def make_env(rank, target_range=RESUME_RANGE):
    def _init():
        return Monitor(NavEnv(target_range=target_range))
    return _init

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,  exist_ok=True)

    print(f"[RESUME] Loading best_model.zip + replay_buffer.pkl")
    print(f"[RESUME] Starting at stage {RESUME_STAGE} ({RESUME_RANGE}m), {RESUME_STEPS:,} steps")

    train_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    eval_env  = SubprocVecEnv([make_env(i) for i in range(4)])

    model = SAC.load(
        os.path.join(SAVE_DIR, "best_model"),
        env=train_env,
        device="auto",
        verbose=1,
        tensorboard_log=LOG_DIR,
    )
    model.load_replay_buffer(os.path.join(SAVE_DIR, "replay_buffer"))
    print(f"[RESUME] Replay buffer loaded: {model.replay_buffer.size()} transitions")

    cb = EvalCurriculumCallback(
        train_env=train_env,
        eval_env=eval_env,
        save_dir=SAVE_DIR,
        log_dir=LOG_DIR,
    )
    # Set stage to RESUME_STAGE so curriculum continues from correct point
    cb._stage = RESUME_STAGE

    rb_cb = ReplayBufferSaveCallback(save_dir=SAVE_DIR, save_freq=200_000, verbose=1)
    callbacks = CallbackList([rb_cb, cb])

    t0 = time.perf_counter()
    model.learn(
        total_timesteps=RESUME_STEPS,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=False,
    )
    elapsed = time.perf_counter() - t0

    model.save(os.path.join(SAVE_DIR, "nav_final"))
    model.save_replay_buffer(os.path.join(SAVE_DIR, "replay_buffer"))
    print(f"\n[DONE] {elapsed/60:.1f} min")
    print(f"  Final model → {SAVE_DIR}/nav_final.zip")

    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
