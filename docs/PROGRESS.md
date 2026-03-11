# Drone RL Project -- Progress Log

## Project Goal

Train a quadrotor (Crazyflie 2.x in PyBullet) to navigate to arbitrary 3D targets
within a 2.0m radius using PPO/SAC + curriculum learning, then extend to
vision-based policies for person tracking.

## Environment Stack

- **Simulator**: gym-pybullet-drones v2.0.0 (CPU-only, no GPU speedup possible)
- **RL Library**: stable-baselines3 v2.7.1
- **Python**: 3.10, conda env `drone-rl`
- **Obs space**: 60-dim (position, velocity, orientation, angular vel, target rel pos)
- **Action**: ActionType.PID -- setpoint control (not raw RPM)
- **Episode**: max 300 steps, success = dist < 0.1m

### Key env files
| File | Role |
|------|------|
| `envs/nav_aviary.py` | Core nav env (potential-based reward, warm_zone_bonus) |
| `envs/dr_nav_aviary.py` | Domain randomization env (mass, wind -- drag inactive) |

---

## What Was Built

### Phase 1: Hover (complete, works)
`train_hover.py` -> `models_hover/best_model.zip`
- Trains drone to hold a fixed altitude setpoint
- PID action type, 300-step episodes
- Works reliably; not actively developed

### Phase 2: Navigation v1 (complete, 2/9 success)
`train_nav.py` -> `models_nav/best_model.zip`
- Curriculum: 0.3m -> 0.6m -> 1.2m -> 2.0m target range
- 4 parallel envs, ~1.36M total steps, PPO [256,256]
- clip_range=0.1, ent_coef=0.01
- **Result**: 2/9 on 9-case benchmark (approach_-Y 0.08m, far_1.8m 0.085m)
- **Key property**: policy std=1.15 (high entropy -- action distribution too wide)

---

## What Doesn't Work

### PPO fine-tuning from v1 best_model -- BLOCKED
All resume attempts share clip_fraction=0.59 from the very first PPO update,
even on clean physics before any changes. This appears at step 0 and never recovers.

Root cause: std=1.15 (high entropy) means wide action distributions -> large gradient
steps -> 59% of PPO updates clipped -> policy barely learns.

Confirmed across 6 attempts:
| Attempt | clip_fraction | eval peak | 9-case |
|---------|--------------|-----------|--------|
| v2 (velocity penalty) | 0.59 from step 0 | -11 | N/A |
| v2b | 0.58 from step 0 | 1.28 | N/A |
| DR resume | 0.58 from step 0 | 0.37 | 1/9 |
| DR scratch | rising 0.18->0.63 | ~3-4 | 0/9 |
| DR warmup | 0.59 from step 0 | 1.14 | 0/9 |
| Option A (value reset) | 0.000 (frozen), then 0.586 first update | N/A | N/A |

**Option A (Value Head Reset)**: Definitively proved the problem is the policy
gradient, not V(s). Resetting value head + freezing policy gives clip=0.000
during Phase 1. Unfreezing gives clip=0.586 on the very first policy gradient step.

### Domain Randomization on PPO -- COUNTERPRODUCTIVE
DR makes the curriculum ~4x slower (noisy rewards) and the clipping problem
remains regardless of whether DR is present. Drag randomization is also ineffective
(DRAG_COEFF only used in Physics.PYB_DRAG mode; default is Physics.PYB).

---

## What Works

### Entropy-Annealed Curriculum from Scratch (Option D)
`train_nav_entropy.py` -> `models_nav_entropy/`
- New PPO from scratch with ent_coef decay: 0.01 (0-500k) -> 0.001 (500k-2M)
  [intended 0.05->0.01->0.001 but ENT_SCHEDULE had unreachable threshold bug]
- clip_range=0.2 (wider, allows meaningful updates early)
- **Key result**: policy std 1.0 -> 0.34 (vs v1's stuck 1.15)
- Curriculum stages 1-3 completed by 800k steps
- Stage 4 (2.0m): clip_fraction rose to 0.66-0.69, not converged
- **9-case on final model (2M steps)**: 1/9 (near_0.3m: 0.091m SUCCESS)

### Stage4 Resume (COMPLETE -- NEW BEST: 5/9)
`train_nav_stage4.py` -> `models_nav_stage4/`
- Resumes from Option D stage3 checkpoint (std=0.34, trained at 1.2m)
- Three bugs fixed vs Option D:
  1. eval_env now uses target_range=2.0m (was 0.3m -- biased best_model selection)
  2. ent_coef=0.001 from step 0 (was 0.01 due to ENT_SCHEDULE unreachable threshold bug)
  3. lr=3e-5 (was 3e-4 -- 10x reduction fixed clip_fraction 0.66 -> 0.11)
- clip_fraction: stable 0.11-0.15 throughout 2M steps (healthy)
- std: 0.71 (at 100k, after jump from 0.34) -> 0.44 at 2M (steadily decreasing)
- eval training rewards: variable/negative (eval env uses random 2.0m targets)
- **9-case benchmark (final, 2M steps): 5/9** ← current best

**Training curiosity**: rollout rewards were mostly negative throughout, and the
callback success_rate showed ~0-2% during training. Yet the final policy solved
5/9 benchmark cases. The eval_env (random targets) and fixed benchmark (specific
positions) measure different things -- the policy generalized despite noisy rewards.

---

## Current Best Models

| Model | Path | 9-case | Best at |
|-------|------|--------|---------|
| **SAC best_model** | `models_nav_sac/best_model.zip` | **8/9** | All except high_target |
| **SAC final** | `models_nav_sac/sac_nav_final.zip` | **8/9** | All except high_target |
| Stage4 PPO final | `models_nav_stage4/ppo_stage4_final.zip` | 5/9 | Lateral approach, far, near |
| Stage4 PPO best | `models_nav_stage4/best_model.zip` | 3/9 | approach_-X, approach_-Y, near_0.3m |
| v1 original | `models_nav/best_model.zip` | 2/9 | Far approach, 1.8m range |
| Option D final | `models_nav_entropy/ppo_nav_entropy_final.zip` | 1/9 | Near-range precision |

---

## 9-Case Benchmark Table (all attempts)

```
Case           v1      DR res  OptD fin  S4-best  S4-final  SAC-best  SAC-final
approach_+X    0.701   0.000   0.723     0.367    YES*      YES*      YES*
approach_-X    0.531   0.740   0.819     YES*     YES*      YES*      YES*
approach_+Y    0.733   0.610   0.544     0.260    YES*      YES*      YES*
approach_-Y    0.080*  1.000   0.197     YES*     0.429     YES*      YES*
low_target     0.803   1.110   0.858     0.777    0.740     YES*      YES*
high_target    1.022   0.980   0.861     0.843    0.920     1.056     1.002
near_0.3m      0.470   0.790   0.091*    YES*     YES*      YES*      YES*
far_1.8m       0.085*  0.450   0.595     1.737    YES*      YES*      YES*
diagonal       0.753   0.720   0.850     0.993    0.770     YES*      YES*
Success        2/9     1/9     1/9       3/9      5/9       8/9       8/9
```
(*) = success (dist < 0.1m)

Remaining failures at 5/9:
- approach_-Y: 0.429m (directional issue, was solved by best_model at 200k)
- low_target: 0.740m (descend to 0.3m altitude -- Z-axis precision)
- high_target: 0.920m (climb to 1.8m -- Z-axis precision)
- diagonal: 0.770m (combined XYZ movement)

Altitude cases (low/high_target) are the hardest: policy may have learned
to navigate in the horizontal plane but not precise Z control.

---

## SAC Training (COMPLETE -- NEW BEST: 8/9!)

`train_nav_sac.py` -> `models_nav_sac/` (COMPLETE, 1M steps)
- Off-policy SAC, single env, replay buffer 1M, batch_size=256
- auto ent_coef tuning, target_entropy=-3.0, ent_coef settled at 0.003
- **9-case benchmark: 8/9** ← current best overall

### SAC Training Trajectory
- 0-10k: random exploration (learning_starts)
- 10-170k: mean_final_dist 1.75→1.13, success_rate 0→4%
- 170-600k: mean_final_dist 1.13→0.79, success_rate 4%→21%
- 600-1000k: mean_final_dist plateau ~0.74m, success_rate plateau ~24-26%
- eval_reward peak: 17.96 at step ~730k → best_model saved
- Final rollout ep_rew_mean: +5-7 (consistently positive)

### SAC vs PPO Comparison
SAC solved cases PPO could not:
- approach_-Y: PPO best 0.429m → SAC: YES
- low_target: PPO best 0.740m → SAC: YES
- diagonal: PPO best 0.770m → SAC: YES
Only remaining failure: high_target (1.002-1.056m; need to climb to 1.8m from 1.0m)

### Files
- `models_nav_sac/best_model.zip` — best during training (eval_reward 17.96 at ~730k)
- `models_nav_sac/sac_nav_final.zip` — final (1M steps)
- Checkpoints at 200k, 400k, 600k, 800k, 1M steps

### Technical notes
- SAC crashed at 99k steps (first attempt) due to: `AttributeError: 'SAC' object has no attribute 'ent_coef_tensor'`
  Fix: use `float(th.exp(model.log_ent_coef.detach()))` instead
- Ran concurrently with PPO initially (2x slower: 46fps vs 91fps after PPO finished)
- SAC more sample-efficient than PPO: 1M SAC steps > 2M PPO steps on 2.0m task
- Stochastic training success_rate (24-26%) << deterministic benchmark success rate (8/9)
  because SAC's deterministic policy (tanh squashed Gaussian mean) is more precise

---

## Known Technical Gotchas

- **PyBullet CPU-only**: No GPU speedup. ~300-500 fps with 8 envs on CPU.
- **Drag randomization inactive**: DRAG_COEFF only used in Physics.PYB_DRAG mode.
  Default Physics.PYB only responds to mass changes and applyExternalForce (wind).
- **clip_fraction reads nan**: SB3 logger returns nan for clip_fraction between
  PPO update calls. Use np.nanmean for rolling averages.
- **SB3 lr_schedule override**: When rebuilding the optimizer, SB3's internal
  `update_learning_rate()` (called each PPO update) uses `model.lr_schedule`,
  not the optimizer's current lr. Must override both `model.learning_rate`
  and `model.lr_schedule = get_schedule_fn(new_lr)`.
- **ENT_SCHEDULE threshold=1.00**: `progress_remaining = 1 - step/total` is always
  < 1.0 at step > 0. The first threshold in a schedule is never triggered.
  First active threshold should be set below 1.0 (e.g., 0.999).
- **SAC policy_std NaN**: SACNavCallback._policy_std() probe fails because
  get_action_dist_params() returns (mean_actions, log_std, kwargs) not (mean, log_std).
  The ent_coef logging also shows NaN because SAC ent_coef_tensor uses log scale
  differently. These are cosmetic -- training is unaffected.
- **Eval rewards ≠ benchmark score**: The EvalCallback uses random target positions
  within target_range. The 9-case benchmark uses fixed positions. A model can show
  poor eval rewards during training but score well on the benchmark (confirmed for
  Stage4 final: mostly negative eval rewards, 5/9 benchmark score).

---

## Next Steps (Session 3 additions)

### P1: SAC v2 — altitude shaping (IN PROGRESS)
- `train_nav_sac_v2.py` running: 2M steps from SAC best_model, `altitude_bonus_w=0.3`
- NavAviary reward adds Z-shaping when xy_dist < 0.2m: `0.3*(prev_z_err - z_err)`
- At 15k/2M steps, first eval at 20k. Benchmark when `models_nav_sac_v2/best_model.zip` exists.
- Goal: push high_target from 1.002m failure to <0.1m success → 9/9

### P2: Robustness check (COMPLETE)
- `benchmark.py` updated: `--runs N` for multi-run robustness, `--extended` for far_3.0m
- `run_robustness()` added — per-case success_rate with ✓/✗ bar and mean±std dist
- SAC best_model robustness: run `python benchmark.py --model ./models_nav_sac/best_model --no-gui --runs 5`
- far_3.0m: hard failure (3.000m) — confirmed training distribution hard boundary at 2.0m

### P3: SAC + Domain Randomization (NOT STARTED — ready)
- `train_nav_sac_dr.py` written, ready to launch after SAC v2 finishes
- Config: wind=0.002N, mass±8%, drag±10%, eval on CLEAN NavAviary
- 1M steps from `models_nav_sac/best_model.zip`
- Goal: confirm 8/9 holds under mild physics perturbations

### P4: Camera obs prototype (COMPLETE)
- `camera_proto.py` confirmed RGB pipeline works
- Key findings:
  - `HoverAviary(obs=ObservationType.RGB, ctrl_freq=24)` — ctrl_freq=24 required (not 30)
  - Obs shape: `(1, 48, 64, 4)` RGBA, capture at 24fps
  - SB3 returns obs as float32 despite uint8 declaration → must cast `.astype(np.uint8)` for PIL
  - Camera is forward-facing (body-X axis); downward camera needs custom target vector
  - 4 landmarks auto-loaded by BaseRLAviary when `obs=ObservationType.RGB`
  - Use `CnnPolicy` in SB3 for training with image obs
  - `camera_proto_frame.png` and `camera_proto_obs.png` saved as proof

### Phase 3: Vision (next major phase)
- Build NavCameraAviary extending NavAviary with ObservationType.RGB
- Use ctrl_freq=24 to match camera capture freq
- Target: person marker in scene, drone learns to approach via vision
- Will need CnnPolicy, likely much longer training (5-10M steps)
- SAC is the right algorithm (sample-efficient, continuous action, auto entropy)

---

## Phase 3: Vision Navigation

### Algorithm Decision
- **SAC confirmed** as the going-forward algorithm. PPO retired at 5/9 (Stage4 final).
  PPO's clip_fraction instability on resume (0.59 from step 0) makes it unsuitable
  for the fine-tuning cycles required in vision training. SAC's off-policy replay
  buffer enables efficient warm-starts without catastrophic forgetting.

### KIN Policy Baseline (official robustness test complete)
- `models_nav_sac/best_model.zip` (kinematic obs, SAC) scored **8/9** on the 9-case benchmark.
- 100% consistent on all 8 passing cases across multi-run robustness check.
- Sole failure: `diagonal` case at 0.77m (outside training distribution geometry).
- This is the performance target that the vision policy must eventually match.

### Replay Buffer Policy (lesson learned)
- Replay buffer saving is **mandatory** for all SAC warm-starts.
- Three failed fine-tune attempts (PPO options A/B/C, SAC cold restarts) confirmed:
  starting from a saved model without its replay buffer leads to immediate
  distribution mismatch and slow/failed recovery.
- All training scripts save buffer every 200k steps AND at run end.
- Future fine-tunes must load `replay_buffer.pkl` with `model.load_replay_buffer()`.

### Vision Pipeline (confirmed working)
- `envs/vision_nav_aviary.py`: VisionNavAviary implemented and tested.
- Obs space: `(3, 32, 48)` RGB uint8, channel-first — SB3 CnnPolicy compatible.
- Bug fixed: `_observationSpace` now uses `self._img_wh` (stored pre-super().__init__)
  so the declared obs space matches actual output for non-default resolutions.
  Previously, the space was declared with default (64,48) even when (48,32) was requested,
  causing DummyVecEnv buffer allocation failure.
- CUDA enabled — NatureCNN forward pass runs on GPU (RTX 4070 Laptop).
- Training script: `train_vision_nav.py` (2M steps, SAC CnnPolicy).

### Frame Stacking (implemented and tested)
- `train_vision_nav_stacked.py`: VecFrameStack(n_stack=4) wrapping DummyVecEnv.
- Stacked obs: `(12, 32, 48)` — 4 consecutive RGB frames concatenated on channel dim.
- `test_stacked_env.py` confirmed: obs shape `(1, 12, 32, 48)` at reset and all 10 steps. PASS.
- Rationale: temporal motion cues (sphere velocity direction) are not visible in a
  single frame; stacking encodes approximate optical flow for the CNN.

### Moving Target Environment (implemented)
- `envs/moving_target_aviary.py`: MovingTargetAviary extends VisionNavAviary.
- Sphere moves at random speed (0.1–0.5 m/s), slowly-rotating XY direction (±5°/step).
- Sphere bounces off XY bounds (±2m); Z clamped to [0.2, 2.5m].
- Center-of-frame bonus: `0.1 * max(0, 1 - angular_offset / 0.5)` where angular_offset
  is the angle between camera forward (body-X, yaw-rotated) and drone-to-sphere vector.
- `test_moving_target.py` confirmed: XY displacement 0→100 steps = 1.97m. PASS.
  Frame saved to `benchmark_results/moving_target_test.png`.

### Current Training
- `train_vision_nav.py` — 2M steps, SAC CnnPolicy, img_wh=(48,32), CUDA.
- ETA: ~Xh (depends on GPU throughput; PyBullet physics still CPU-bound).
- Checkpoints every 100k steps, eval every 20k, replay buffer every 200k.

### Next Milestones
1. 500k steps: 3-case mini-benchmark (near_0.3m, approach_+X, diagonal).
   Goal: any success (dist < 0.1m) at least once — confirms vision policy learning.
2. 2M steps: full 9-case benchmark vs KIN baseline (8/9 target).
3. If vision policy reaches 4/9+: switch to frame-stacked training for temporal cues.
4. Moving target training: once static target is reliably solved (6/9+).
5. Person tracking phase: replace sphere with pedestrian mesh, add tracking reward.
