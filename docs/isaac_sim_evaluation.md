# Isaac Sim / OmniDrones Evaluation
**Date:** 2026-03-10
**Question:** Should we upgrade to Isaac Sim (OmniDrones) for the vision/person-tracking phase, or stay in PyBullet?

---

## Research Summary

### 1. OmniDrones — Camera Observations

OmniDrones' documentation notes that Isaac Sim 2023.1.0 introduced "new sensors," suggesting camera/RGB observation support is available through Isaac Sim's native sensor API. However, the existing OmniDrones task suite (Hover, Track, FlyThrough, PayloadHover, Formation, etc.) appears to use **state-vector observations only** — none of the 17 documented tasks explicitly use pixel/camera observations. Adding camera obs would require custom implementation using Isaac Sim's `rep.create.camera()` Replicator API, which is non-trivial.

**Verdict:** Camera obs are possible but not included out-of-the-box. Custom implementation required.

### 2. Pedestrian / Moving Target Agents

No pedestrian or moving-target environments are present in OmniDrones. The `Track` task tracks a pre-defined trajectory, not an independent moving agent. Isaac Sim does support importing animated humans (via Omniverse USD assets and the People Simulation extension), but integrating these as RL-visible objects requires non-trivial Isaac Sim scene scripting.

**Verdict:** No pedestrian support in OmniDrones. Building it would require significant Isaac Sim scene authoring work.

### 3. GPU / CUDA Requirements

NVIDIA's official Isaac Sim requirements (as of early 2026):

| Tier    | GPU             | VRAM   |
|---------|-----------------|--------|
| Minimum | RTX 4080        | 16 GB  |
| Ideal   | RTX PRO 6000    | 48 GB  |

Key constraints:
- **16 GB VRAM minimum** — the specification page explicitly warns that GPUs with less than 16 GB VRAM "may be insufficient."
- Isaac Sim requires driver 580.88+ (Windows) / 580.65.06+ (Linux).
- Data-center GPUs without RT Cores (A100, H100) are explicitly unsupported.
- Isaac Sim runs on Windows 10/11 and Ubuntu 22/24; **no macOS support**.
- Training with Isaac Lab requires **additional RAM and VRAM** beyond the base Isaac Sim minimums.

**Our GPU: RTX 4070 Laptop (8 GB VRAM)**
The RTX 4070 Laptop has 8 GB VRAM — half the stated minimum. It has RT Cores (required) but falls well below the VRAM floor. Isaac Sim may launch for basic visualization, but running a training loop with multiple drone environments and camera renders simultaneously is very likely to OOM or require extreme scene simplification.

### 4. SB3 / Gym Compatibility

Isaac Lab (the RL training layer above Isaac Sim) officially supports SB3 as one of four supported frameworks. However, SB3 in Isaac Lab has **no vectorized training support and no distributed training support** — which eliminates the primary benefit of GPU-accelerated simulation (parallel envs). The native Isaac Lab workflows use RSL-RL or RL-Games for vectorized GPU rollouts.

OmniDrones itself uses TorchRL / tensordict as its RL interface, not gymnasium. Wrapping OmniDrones envs as standard gymnasium environments is possible but requires a compatibility shim, and is not officially supported.

**Verdict:** SB3 works but loses all vectorization benefits. Native GPU training requires switching from SB3 to RSL-RL/RL-Games.

### 5. Effort to Port NavAviary Reward Structure

Porting our reward structure (potential-based distance shaping + warm zone bonus + crash penalty + center-of-frame bonus) to OmniDrones would require:
1. Subclassing `IsaacEnv` (OmniDrones base class) — different API from gymnasium/BaseRLAviary.
2. Rewriting `_computeReward`, `_computeObs`, `_computeTerminated` in Isaac Sim's tensor-based API (all observations/rewards must be torch tensors on GPU).
3. Adding a colored sphere target using Isaac Sim's Replicator or USD API.
4. Camera obs: implement Isaac Sim camera sensor + channel-first conversion for NatureCNN.
5. Integration testing — Isaac Sim has a long startup time (~2 min cold start) making iteration slow.

Estimated effort: **2–3 weeks** to reach feature parity with current VisionNavAviary.

---

## Recommendation: Stay in PyBullet

**Recommendation: Do NOT upgrade to Isaac Sim at this stage. Continue in PyBullet.**

### Reasons

**GPU blocker (hard constraint):**
The RTX 4070 Laptop has 8 GB VRAM. Isaac Sim's minimum spec is 16 GB. This is not a soft recommendation — the requirement page explicitly flags sub-16GB as potentially insufficient even for basic rendering, before any RL training load. Running multi-env training with camera renders would almost certainly OOM. This single constraint makes Isaac Sim impractical on current hardware.

**No camera obs out-of-the-box:**
OmniDrones provides only state-vector observations. The entire point of the vision phase is pixel-based policy learning. We would spend 2–3 weeks building camera infrastructure in an unfamiliar framework before writing a single line of RL code.

**No moving target / pedestrian support:**
We have already implemented MovingTargetAviary in PyBullet in one session. The Isaac Sim equivalent would require USD scene authoring and the People Simulation extension — a multi-day effort with steep learning curve.

**SB3 vectorization limitation:**
The one compelling reason to move to Isaac Sim is GPU-parallel physics rollouts. But SB3 in Isaac Lab has no vectorized training support, so we cannot use it there. Switching to RSL-RL/RL-Games would require rewriting all callbacks, curriculum logic, and replay buffer management.

**PyBullet is sufficient for Phase 3:**
Phase 3 goals are: (a) train a vision policy to reliably navigate to a static target, (b) extend to a moving sphere, (c) build toward person tracking. PyBullet at ~50–80 fps single env is slow but workable for 2M-step SAC training. The CPU bottleneck is the physics, not GPU — and NatureCNN forward pass already runs on our RTX 4070 Laptop GPU via CUDA. We are GPU-accelerating the part that benefits from it.

**Isaac Sim is worth revisiting when:**
- Hardware upgrade to RTX 4090 (24 GB) or RTX 5080 (16 GB+).
- Vision policy is mature and we need >10x speedup for large-scale curriculum.
- Moving to state-of-the-art sim-to-real transfer work that needs photorealistic rendering.
- Team has bandwidth to invest 2–3 weeks in framework migration.

### Decision Table

| Factor                      | PyBullet (current) | Isaac Sim / OmniDrones |
|-----------------------------|-------------------|------------------------|
| VRAM requirement            | ~1 GB (CPU sim)   | 16 GB minimum          |
| Works on RTX 4070 Laptop    | Yes               | Likely OOM             |
| Camera obs built-in         | Yes (VisionNavAviary) | No — custom build needed |
| Moving target env           | Yes (built)       | No — custom build needed |
| SB3 compatibility           | Full              | Limited (no vecenv)    |
| Windows support             | Yes               | Yes (but slower)       |
| Porting effort              | 0                 | ~2–3 weeks             |
| Sim speed (single env)      | ~50–80 fps        | ~500–5000 fps (GPU)*   |
| Parallel envs (SB3)         | DummyVecEnv (CPU) | Not supported in SB3   |

*GPU speedup only available with non-SB3 frameworks (RSL-RL, RL-Games).

**Conclusion:** Isaac Sim would be a significant capability upgrade for a future hardware cycle. For the current RTX 4070 Laptop + SB3 + vision navigation phase, stay in PyBullet.
