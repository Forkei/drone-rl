"""
100k fine-tune attempt for 9/9: bias target sampling toward high vertical targets.

Loads nav_final + replay_buffer_stage3.pkl.
40% of episodes: target directly above drone (z+0.5 to z+1.5m)
60% of episodes: normal sphere sampling at 2.0m range.

Expected: high_target improves without regressing far_1.8m (replay buffer anchor).
Run time: ~3 min at 550 fps.
"""

import os, sys, time
sys.path.insert(0, r"C:\Users\forke\Documents\Drones\PyBullet1")
os.chdir(r"C:\Users\forke\Documents\Drones\PyBullet1")

import numpy as np
import mujoco
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.mujoco.nav_env import NavEnv, MAX_STEPS_NAV

SAVE_DIR = "./models_nav_mujoco"
N_ENVS   = 8


class HighBiasNavEnv(NavEnv):
    """NavEnv with 40% probability of pure-vertical high targets."""

    def reset(self, *, seed=None, options=None):
        self._consec_goal = 0
        super(NavEnv, self).reset(seed=seed)   # gym.Env seed handling
        mujoco.mj_resetData(self.model, self.data)

        drone_home = np.array([
            self.np_random.uniform(-0.5, 0.5),
            self.np_random.uniform(-0.5, 0.5),
            self.np_random.uniform(0.5, 1.2),
        ])
        self.data.qpos[:3] = drone_home
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qvel[:] = 0.0

        if self.np_random.random() < 0.4:
            # High vertical target
            dz = self.np_random.uniform(0.5, 1.5)
            target = drone_home + np.array([0.0, 0.0, dz])
        else:
            # Normal sphere sampling
            min_dist = max(0.1, self.target_range * 0.25)
            direction = self.np_random.uniform(-1, 1, size=3)
            direction /= np.linalg.norm(direction) + 1e-8
            dist = self.np_random.uniform(min_dist, self.target_range)
            target = drone_home + direction * dist

        target[2] = np.clip(target[2], 0.3, 2.5)
        self._set_target(target)

        mujoco.mj_forward(self.model, self.data)
        self._prev_dist  = np.linalg.norm(self._target_pos - drone_home)
        self._step_count = 0
        return self._get_obs(), {}


def make_env(rank):
    def _init():
        return Monitor(HighBiasNavEnv(target_range=2.0))
    return _init


def quick_benchmark(model, label):
    """Run high_target and far_1.8m cases, print results."""
    from envs.mujoco.nav_env import NavEnv
    env = NavEnv(target_range=2.0)

    cases = {
        "high_target": {"start": [0.0, 0.0, 1.0], "target": [0.0, 0.0, 1.8]},
        "far_1.8m":    {"start": [1.8, 0.0, 1.0], "target": [0.0, 0.0, 1.0]},
    }

    for name, case in cases.items():
        results = []
        for _ in range(3):
            mujoco.mj_resetData(env.model, env.data)
            start  = np.array(case["start"])
            target = np.array(case["target"])
            env.data.qpos[:3] = start
            env.data.qpos[3:7] = [1, 0, 0, 0]
            env.data.qvel[:] = 0.0
            env._set_target(target)
            mujoco.mj_forward(env.model, env.data)
            env._prev_dist  = float(np.linalg.norm(target - start))
            env._step_count = 0
            env._consec_goal = 0
            obs = env._get_obs()

            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc
            results.append((info["dist_to_target"], info["success"], info["steps"] if "steps" in info else 0))

        ok = sum(r[1] for r in results)
        mean_dist = np.mean([r[0] for r in results])
        print(f"  [{label}] {name}: {ok}/3  dist={mean_dist:.4f}m")
    env.close()


def main():
    print("[FINETUNE] Loading nav_final + replay_buffer_stage3")
    train_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])

    model = SAC.load(
        os.path.join(SAVE_DIR, "nav_final"),
        env=train_env,
        device="auto",
        verbose=1,
    )
    rb_path = os.path.join(SAVE_DIR, "replay_buffer_stage3")
    if os.path.exists(rb_path + ".pkl"):
        model.load_replay_buffer(rb_path)
        print(f"  Replay buffer loaded: {model.replay_buffer.size()} transitions")
    else:
        print("  WARNING: replay_buffer_stage3.pkl not found, using empty buffer")

    print("\n[BEFORE fine-tune]")
    quick_benchmark(model, "before")

    t0 = time.perf_counter()
    model.learn(
        total_timesteps=100_000,
        reset_num_timesteps=False,
        progress_bar=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nFine-tune done in {elapsed/60:.1f} min")

    model.save(os.path.join(SAVE_DIR, "nav_finetuned_high"))
    model.save_replay_buffer(os.path.join(SAVE_DIR, "replay_buffer_finetuned"))

    print("\n[AFTER fine-tune]")
    quick_benchmark(model, "after")

    train_env.close()
    print("\n[DONE] nav_finetuned_high.zip saved")


if __name__ == "__main__":
    main()
