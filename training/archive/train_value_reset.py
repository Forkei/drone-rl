"""
Option A: Value Head Reset.

Loads v1 best_model, resets value head weights (mlp_extractor.value_net +
value_net), freezes the policy network (mlp_extractor.policy_net +
action_net + log_std), then trains only the value head at lr=1e-5 for
50k steps.

Go/no-go check at 50k steps (clip_fraction from last PPO update):
    < 0.3  -> Phase 2: unfreeze all params, lr=3e-5, 200k more steps
    >= 0.5 -> STOP -- go to Option D

Key metrics logged to TensorBoard:
    vr/clip_fraction     — from PPO trainer (first-class)
    vr/policy_std        — exp(log_std).mean()
    vr/eval_reward       — clean eval
    vr/phase             — 0=value-only, 1=full
    vr/success_rate
    vr/mean_final_dist

Usage:
    python train_value_reset.py [--resume ./models_nav/best_model]
"""

import argparse
import os
import numpy as np
from collections import deque

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor

from envs.nav_aviary import NavAviary


PHASE1_STEPS = 50_000   # value-head only
PHASE2_STEPS = 200_000  # full network
LR_PHASE1    = 1e-5
LR_PHASE2    = 3e-5
CLIP_GO      = 0.3      # below this at 50k -> Phase 2
CLIP_NOGO    = 0.5      # above this at 50k -> STOP


# ── helpers ───────────────────────────────────────────────────────────────────

def reset_value_head(model):
    """Reset value network weights to kaiming uniform (PyTorch default)."""
    print("\n[ValueReset] Resetting value head weights...")
    for layer in model.policy.mlp_extractor.value_net:
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    model.policy.value_net.reset_parameters()
    print("[ValueReset] Done — mlp_extractor.value_net + value_net reset.\n")


def freeze_policy_net(model):
    """Freeze policy network, action_net, log_std. Keep value params trainable."""
    for param in model.policy.mlp_extractor.policy_net.parameters():
        param.requires_grad = False
    for param in model.policy.action_net.parameters():
        param.requires_grad = False
    model.policy.log_std.requires_grad = False

    trainable = [p for p in model.policy.parameters() if p.requires_grad]
    n_frozen  = sum(p.numel() for p in model.policy.parameters() if not p.requires_grad)
    n_train   = sum(p.numel() for p in trainable)
    print(f"[ValueReset] Policy frozen. Frozen params: {n_frozen:,} | Trainable: {n_train:,}")
    return trainable


def unfreeze_all(model):
    """Unfreeze all parameters for Phase 2."""
    for param in model.policy.parameters():
        param.requires_grad = True
    model.policy.log_std.requires_grad = True
    n = sum(p.numel() for p in model.policy.parameters())
    print(f"\n[ValueReset] All params unfrozen for Phase 2. Total: {n:,}")


def set_lr(model, lr):
    """Update optimizer learning rate."""
    for pg in model.policy.optimizer.param_groups:
        pg['lr'] = lr


def rebuild_optimizer(model, params, lr):
    """Rebuild Adam optimizer with given params and lr. Clears old Adam state."""
    model.policy.optimizer = th.optim.Adam(params, lr=lr, eps=1e-5)
    print(f"[ValueReset] Optimizer rebuilt — lr={lr}, {len(params)} param tensors")


# ── callback ──────────────────────────────────────────────────────────────────

class ValueResetCallback(BaseCallback):
    """
    Monitors clip_fraction, policy std, and triggers Phase 2 at step 50k.

    Logs:
        vr/clip_fraction, vr/clip_fraction_25k  — rolling window
        vr/policy_std                           — action std (exp(log_std).mean())
        vr/eval_reward                          — from EvalCallback
        vr/phase                                — 0=value-only, 1=full
        vr/success_rate, vr/mean_final_dist
    """

    def __init__(self, train_env, save_dir, window=100, check_freq=2048, verbose=1):
        super().__init__(verbose)
        self.train_env    = train_env
        self.save_dir     = save_dir
        self.window       = window
        self.check_freq   = check_freq
        self._successes   = deque(maxlen=window)
        self._ep_dists    = deque(maxlen=window)
        self._clip_hist   = deque(maxlen=15)   # ~25k steps at check_freq=2048
        self._phase       = 0
        self._phase2_done = False
        self.last_eval_reward = -np.inf
        # Report clip_fraction at these local steps
        self._report_at   = {25_000, 50_000}
        self._reported    = set()

    def notify_eval_reward(self, r):
        self.last_eval_reward = r

    @property
    def _local_steps(self):
        return self.model.num_timesteps

    def _policy_std(self):
        return float(th.exp(self.model.policy.log_std).mean().detach())

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._successes.append(float(info.get("success", 0.0)))
                if "dist_to_target" in info:
                    self._ep_dists.append(info["dist_to_target"])

        local = self._local_steps
        clip_frac = self.model.logger.name_to_value.get("train/clip_fraction", float("nan"))
        if not np.isnan(clip_frac):
            self._clip_hist.append(clip_frac)

        # ── report checkpoints ─────────────────────────────────────────────
        for milestone in list(self._report_at - self._reported):
            if local >= milestone:
                self._reported.add(milestone)
                cf_now = clip_frac if not np.isnan(clip_frac) else float(np.nanmean(self._clip_hist) if self._clip_hist else float("nan"))
                std    = self._policy_std()
                print(f"\n{'='*60}")
                print(f"[A] CHECKPOINT at local step {local:,}  (milestone {milestone:,})")
                print(f"    clip_fraction : {cf_now:.4f}")
                print(f"    policy_std    : {std:.4f}")
                print(f"    eval_reward   : {self.last_eval_reward:.3f}")

                if milestone == 50_000 and self._phase == 0:
                    # Go/no-go decision
                    cf_check = float(np.nanmean(self._clip_hist)) if self._clip_hist else cf_now
                    if cf_check < CLIP_GO:
                        print(f"    DECISION: clip={cf_check:.3f} < {CLIP_GO} -> GO -- starting Phase 2")
                    elif cf_check >= CLIP_NOGO:
                        print(f"    DECISION: clip={cf_check:.3f} >= {CLIP_NOGO} -> STOP -- go to Option D")
                        print(f"{'='*60}\n")
                        return False   # stop training
                    else:
                        print(f"    DECISION: clip={cf_check:.3f} in [{CLIP_GO},{CLIP_NOGO}) -- marginal, continuing Phase 2")
                    print(f"{'='*60}\n")

                    # Enter Phase 2
                    self._enter_phase2()
                else:
                    print(f"{'='*60}\n")

        if self.n_calls % self.check_freq != 0 or len(self._successes) < 10:
            return True

        # ── regular metrics ────────────────────────────────────────────────
        success_rate = float(np.mean(self._successes))
        mean_dist    = float(np.mean(self._ep_dists)) if self._ep_dists else float("nan")
        rolling_clip = float(np.nanmean(self._clip_hist)) if self._clip_hist else float("nan")
        std          = self._policy_std()

        self.logger.record("vr/clip_fraction",    clip_frac)
        self.logger.record("vr/clip_fraction_25k", rolling_clip)
        self.logger.record("vr/policy_std",        std)
        self.logger.record("vr/eval_reward",       self.last_eval_reward)
        self.logger.record("vr/phase",             self._phase)
        self.logger.record("vr/success_rate",      success_rate)
        self.logger.record("vr/mean_final_dist",   mean_dist)

        std_warn = ""
        if std > 0.8:
            std_warn = f" [STD HIGH {std:.2f}]"
        clip_warn = ""
        if not np.isnan(rolling_clip) and rolling_clip > 0.3:
            clip_warn = f" [CLIP {rolling_clip:.2f}]"

        phase_str = "P1-value-only" if self._phase == 0 else "P2-full"
        if self.verbose:
            print(f"[A-{phase_str}] local={local:,} | "
                  f"eval={self.last_eval_reward:.2f} | "
                  f"dist={mean_dist:.3f}m | "
                  f"suc={success_rate:.2f} | "
                  f"clip={clip_frac:.3f} | "
                  f"std={std:.3f}"
                  f"{clip_warn}{std_warn}")

        return True

    def _enter_phase2(self):
        self._phase = 1
        unfreeze_all(self.model)
        rebuild_optimizer(
            self.model,
            [p for p in self.model.policy.parameters() if p.requires_grad],
            lr=LR_PHASE2,
        )


class EvalVRCallback(EvalCallback):
    def __init__(self, vr_cb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vr_cb = vr_cb

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.last_mean_reward != -np.inf:
            self.vr_cb.notify_eval_reward(self.last_mean_reward)
        return result


# ── training ──────────────────────────────────────────────────────────────────

def train(
    total_timesteps: int = PHASE1_STEPS + PHASE2_STEPS,
    n_envs:          int = 8,
    save_dir:        str = "./models_value_reset",
    log_dir:         str = "./logs_value_reset",
    resume_path:     str = "./models_nav/best_model",
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    train_env = make_vec_env(
        lambda: Monitor(NavAviary(target_range=2.0)), n_envs=n_envs
    )
    eval_env = Monitor(NavAviary(target_range=2.0))

    if not os.path.isfile(resume_path + ".zip"):
        raise FileNotFoundError(f"Not found: {resume_path}.zip")

    print(f"[INFO] Loading {resume_path}.zip")
    model = PPO.load(resume_path, env=train_env,
                     tensorboard_log=log_dir, verbose=1,
                     device="cpu")

    # ── Phase 1 setup ─────────────────────────────────────────────────────
    reset_value_head(model)
    trainable = freeze_policy_net(model)
    rebuild_optimizer(model, trainable, lr=LR_PHASE1)

    # Print baseline std before any training
    std0 = float(th.exp(model.policy.log_std).mean().detach())
    print(f"[INFO] Baseline policy std (before training): {std0:.4f}")
    print(f"[INFO] Phase 1: value-head only, {PHASE1_STEPS:,} steps at lr={LR_PHASE1}")
    print(f"[INFO] Phase 2 (if go): full network, {PHASE2_STEPS:,} steps at lr={LR_PHASE2}")
    print(f"[INFO] Go/no-go at 50k: clip<{CLIP_GO} -> go, clip>={CLIP_NOGO} -> STOP\n")

    vr_cb = ValueResetCallback(
        train_env=train_env,
        save_dir=save_dir,
        window=100,
        check_freq=2048,
        verbose=1,
    )
    callbacks = CallbackList([
        vr_cb,
        EvalVRCallback(
            vr_cb=vr_cb,
            eval_env=eval_env,
            best_model_save_path=save_dir,
            log_path=log_dir,
            eval_freq=max(10_000 // n_envs, 1),  # more frequent evals in Phase 1
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(50_000 // n_envs, 1),
            save_path=save_dir,
            name_prefix="ppo_vr",
        ),
    ])

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=True,  # fresh step counter
    )

    final = os.path.join(save_dir, "ppo_vr_final")
    model.save(final)
    print(f"\n[INFO] Final model -> {final}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=PHASE1_STEPS + PHASE2_STEPS)
    p.add_argument("--n-envs",    type=int, default=8)
    p.add_argument("--save-dir",  default="./models_value_reset")
    p.add_argument("--log-dir",   default="./logs_value_reset")
    p.add_argument("--resume",    default="./models_nav/best_model")
    args = p.parse_args()

    train(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        resume_path=args.resume,
    )
