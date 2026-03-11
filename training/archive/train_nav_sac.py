"""
SAC Navigation Training.

Off-policy alternative to PPO for 2.0m target range.
SAC is more sample-efficient and handles continuous control better
when the task requires precise long-range navigation.

Key differences vs PPO:
- Replay buffer (1M): learns from all past experience, not just current rollout
- Automatic entropy tuning: target_entropy=-3 (=-action_dim) tunes ent_coef
- No clip_range: no PPO-style clipping problem
- Slower per-step (gradient on each step) but better sample efficiency

Start from scratch OR resume from a PPO checkpoint (weights compatible).

Usage:
    # From scratch
    python train_nav_sac.py

    # Resume from PPO stage3 checkpoint
    python train_nav_sac.py --resume-ppo ./models_nav_entropy/nav_ent_stage3_1.2m

    # Longer run
    python train_nav_sac.py --timesteps 2000000
"""

import argparse
import os
import numpy as np
from collections import deque

import torch as th
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor

from envs.nav_aviary import NavAviary


# -- callback ------------------------------------------------------------------

class SACNavCallback(BaseCallback):
    """
    Monitors SAC training: critic loss, actor loss, ent_coef, policy_std, success_rate.

    TensorBoard metrics:
        sac/policy_std, sac/ent_coef
        sac/success_rate, sac/mean_final_dist
        sac/eval_reward
    """

    def __init__(self, save_dir, window=500, check_freq=5000, verbose=1):
        super().__init__(verbose)
        self.save_dir   = save_dir
        self.window     = window
        self.check_freq = check_freq
        self._successes = deque(maxlen=window)
        self._ep_dists  = deque(maxlen=window)
        self.last_eval_reward = -np.inf
        self._diag_100k = False
        self._diag_500k = False

    def notify_eval_reward(self, r):
        self.last_eval_reward = r

    def _policy_std(self):
        # SAC uses a diagonal Gaussian actor -- log_std is state-dependent
        # Use mean over a fixed probe state (zeros) as a rough indicator
        try:
            with th.no_grad():
                probe = th.zeros(1, self.model.observation_space.shape[0],
                                 device=self.model.device)
                dist = self.model.actor.get_action_dist_params(probe)
                # dist is (mean, log_std) for SAC DiagGaussian
                log_std = dist[1] if isinstance(dist, tuple) else None
                if log_std is not None:
                    return float(th.exp(log_std).mean())
        except Exception:
            pass
        return float("nan")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._successes.append(float(info.get("success", 0.0)))
                if "dist_to_target" in info:
                    self._ep_dists.append(info["dist_to_target"])

        step = self.model.num_timesteps

        # Diagnostic at 100k
        if not self._diag_100k and step >= 100_000:
            self._diag_100k = True
            std = self._policy_std()
            try:
                ent_coef = float(th.exp(self.model.log_ent_coef.detach()))
            except Exception:
                try:
                    ent_coef = float(self.model.ent_coef_tensor.exp())
                except Exception:
                    ent_coef = float("nan")
            print(f"\n{'='*55}")
            print(f"[SAC] DIAGNOSTIC at step {step:,}")
            print(f"  policy_std (probe)  : {std:.4f}")
            print(f"  ent_coef (current)  : {ent_coef:.5f}")
            print(f"  eval_reward         : {self.last_eval_reward:.3f}")
            suc = float(np.mean(self._successes)) if self._successes else 0.0
            print(f"  success_rate (500ep): {suc:.3f}")
            print(f"{'='*55}\n")

        if self.n_calls % self.check_freq != 0 or len(self._successes) < 20:
            return True

        success_rate = float(np.mean(self._successes))
        mean_dist    = float(np.mean(self._ep_dists)) if self._ep_dists else float("nan")
        std          = self._policy_std()

        try:
            ent_coef = float(th.exp(self.model.log_ent_coef.detach()))
        except Exception:
            try:
                ent_coef = float(self.model.ent_coef_tensor.exp())
            except Exception:
                ent_coef = float("nan")

        self.logger.record("sac/policy_std",      std)
        self.logger.record("sac/ent_coef",        ent_coef)
        self.logger.record("sac/success_rate",    success_rate)
        self.logger.record("sac/mean_final_dist", mean_dist)
        self.logger.record("sac/eval_reward",     self.last_eval_reward)

        if self.verbose:
            print(f"[SAC] step={step:,} | eval={self.last_eval_reward:.2f} | "
                  f"suc={success_rate:.2f}/0.60 | dist={mean_dist:.3f}m | "
                  f"std={std:.3f} | ent={ent_coef:.4f}")

        return True


class EvalSACCallback(EvalCallback):
    def __init__(self, sac_cb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sac_cb = sac_cb

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.last_mean_reward != -np.inf:
            self.sac_cb.notify_eval_reward(self.last_mean_reward)
        return result


# -- training ------------------------------------------------------------------

def transfer_ppo_to_sac(ppo_path: str, sac_model):
    """
    Transfer PPO actor weights to SAC actor.

    PPO policy_net -> SAC actor latent_pi network (same [256,256] arch)
    PPO action_net -> SAC actor mu (mean) network
    PPO log_std    -> SAC uses state-dependent log_std (different arch; skip)

    Best-effort: only transfers if shapes match exactly.
    """
    print(f"[Transfer] Loading PPO weights from {ppo_path}.zip")
    try:
        ppo = PPO.load(ppo_path, device="cpu")
    except Exception as e:
        print(f"[Transfer] WARNING: could not load PPO model: {e}. Training SAC from scratch.")
        return

    transferred = 0
    skipped     = 0

    # Map: PPO param name -> SAC param name
    ppo_state  = dict(ppo.policy.named_parameters())
    sac_state  = dict(sac_model.policy.named_parameters())

    # PPO: mlp_extractor.policy_net.{0,2}.{weight,bias}
    #      action_net.{weight,bias}
    # SAC: actor.latent_pi.{0,2}.{weight,bias}
    #      actor.mu.{weight,bias}
    name_map = {}
    for k in ppo_state:
        if k.startswith("mlp_extractor.policy_net."):
            sac_key = k.replace("mlp_extractor.policy_net.", "actor.latent_pi.")
            name_map[k] = sac_key
        elif k.startswith("action_net."):
            sac_key = k.replace("action_net.", "actor.mu.")
            name_map[k] = sac_key

    with th.no_grad():
        for ppo_key, sac_key in name_map.items():
            if sac_key in sac_state:
                if ppo_state[ppo_key].shape == sac_state[sac_key].shape:
                    sac_state[sac_key].copy_(ppo_state[ppo_key])
                    transferred += 1
                else:
                    skipped += 1
            else:
                skipped += 1

    print(f"[Transfer] Transferred {transferred} param tensors, skipped {skipped}")


def train(
    total_timesteps: int = 1_000_000,
    save_dir:        str = "./models_nav_sac",
    log_dir:         str = "./logs_nav_sac",
    resume_ppo:      str = None,
    target_range:    float = 2.0,
    learning_rate:   float = 3e-4,
    buffer_size:     int   = 1_000_000,
    batch_size:      int   = 256,
    learning_starts: int   = 10_000,
    target_entropy:  float = -3.0,   # = -action_dim for 3D actions
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    train_env = Monitor(NavAviary(target_range=target_range))
    eval_env  = Monitor(NavAviary(target_range=target_range))

    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        learning_starts=learning_starts,
        target_entropy=target_entropy,    # auto entropy tuning
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256]),
        device="cpu",
        seed=42,
    )

    if resume_ppo:
        transfer_ppo_to_sac(resume_ppo, model)

    print(f"\n[INFO] SAC Navigation Training")
    print(f"[INFO] target_range={target_range}m  lr={learning_rate}  batch={batch_size}")
    print(f"[INFO] buffer_size={buffer_size:,}  learning_starts={learning_starts:,}")
    print(f"[INFO] target_entropy={target_entropy} (auto ent_coef tuning)")
    print(f"[INFO] eval_env at target_range={target_range}m (fixed)")
    if resume_ppo:
        print(f"[INFO] PPO actor weights transferred from {resume_ppo}")
    print()

    sac_cb = SACNavCallback(
        save_dir=save_dir,
        window=500,
        check_freq=5000,
        verbose=1,
    )
    callbacks = CallbackList([
        sac_cb,
        EvalSACCallback(
            sac_cb=sac_cb,
            eval_env=eval_env,
            best_model_save_path=save_dir,
            log_path=log_dir,
            eval_freq=20_000,       # every 20k steps (SAC is single-env)
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=200_000,
            save_path=save_dir,
            name_prefix="sac_nav",
        ),
    ])

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=True,
    )

    final = os.path.join(save_dir, "sac_nav_final")
    model.save(final)
    print(f"\n[INFO] Final model -> {final}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps",       type=int,   default=1_000_000)
    p.add_argument("--save-dir",                    default="./models_nav_sac")
    p.add_argument("--log-dir",                     default="./logs_nav_sac")
    p.add_argument("--resume-ppo",                  default=None,
                   help="Path to PPO checkpoint to transfer actor weights from")
    p.add_argument("--target-range",    type=float, default=2.0)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--batch-size",      type=int,   default=256)
    p.add_argument("--target-entropy",  type=float, default=-3.0)
    args = p.parse_args()

    train(
        total_timesteps=args.timesteps,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        resume_ppo=args.resume_ppo,
        target_range=args.target_range,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        target_entropy=args.target_entropy,
    )
