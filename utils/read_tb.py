"""Read TensorBoard event file and extract key metrics."""
import sys
import os

log_dir = sys.argv[1] if len(sys.argv) > 1 else './logs_nav_stage4/PPO_1'

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(log_dir)
    ea.Reload()

    tags = ea.Tags()['scalars']
    print("Available tags:", tags)

    want = ['train/clip_fraction', 's4/clip_fraction_25k', 's4/policy_std',
            's4/success_rate', 's4/eval_reward', 's4/mean_final_dist']

    for tag in want:
        if tag in tags:
            events = ea.Scalars(tag)
            # Print last 10 values
            recent = events[-10:]
            print(f"\n{tag}:")
            for e in recent:
                print(f"  step={e.step:>8,}  value={e.value:.4f}")
        else:
            print(f"\n{tag}: NOT FOUND")

except ImportError:
    # Fallback: parse raw bytes
    print("tensorboard not available, trying raw parse")
    import glob
    files = glob.glob(os.path.join(log_dir, 'events.out.*'))
    print(f"Found event files: {files}")
    print("Install tensorboard: pip install tensorboard")
