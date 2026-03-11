# Session Summary

## Done
- Created conda env `drone-rl` (Python 3.10) with gym-pybullet-drones v2.0, SB3 v2.7.1, PyTorch, gymnasium
- Trained PPO hover policy (500k steps, 4 parallel envs, 17 min on CPU)
- Best eval reward: **472.88 ± 0.49 / ~960 max** — drone reliably hovers 3–12 cm from target
- Visualized policy in PyBullet GUI, confirmed working

## Files
- `train_hover.py` — PPO training script
- `visualize_hover.py` — GUI rollout + trajectory/reward plots
- `plot_results.py` — post-training curves from logs
- `monitor_training.py` — live reward monitor during training
- `quick_test.py` — headless sanity check
- `models/best_model.zip` — trained policy
- `results.png` — training curves

## Key technical note
Obs shape is `(1, 72)`, not `(12,)` — position is `obs[0, :3]`. Action is normalized RPM `[-1, 1]` shape `(1, 4)`.

## Main limitation
Using `ActionType.RPM` (raw rotor speeds) — drone wobbles visibly because there's no inner stabilization loop and the reward doesn't penalize velocity. It works but isn't smooth.

## Obvious next step
Retrain with `ActionType.PID` — policy outputs thrust + attitude rate targets, built-in PID handles stabilization. Should converge faster and fly much cleaner. One-line change in the training script.
