from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys, os

log_dir = './logs_vision_nav/SAC_2'
if not os.path.exists(log_dir):
    print("SAC_2 not found"); sys.exit(0)

ea = EventAccumulator(log_dir)
ea.Reload()
tags = ea.Tags().get('scalars', [])
print("Tags:", tags[:10])

try:
    s = ea.Scalars('rollout/ep_rew_mean')
    first, last = s[0], s[-1]
    elapsed = last.wall_time - first.wall_time
    fps = (last.step - first.step) / elapsed if elapsed > 0 else 0
    step = last.step
    eta_hr = (2_000_000 - step) / fps / 3600 if fps > 0 else 999
    print(f"step={step:,}  fps={fps:.1f}  ETA={eta_hr:.1f}h  rew={last.value:.3f}")
except Exception as e:
    print(f"rollout error: {e}")

for tag in ['vis/fps', 'vis/success_rate', 'vis/mean_final_dist']:
    try:
        v = ea.Scalars(tag)[-1]
        print(f"  {tag}: {v.value:.4f} @ step {v.step:,}")
    except:
        pass
