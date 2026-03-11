import numpy as np
import sys

path = sys.argv[1] if len(sys.argv) > 1 else './logs_nav_stage4/evaluations.npz'
d = np.load(path)
print('timesteps:', d['timesteps'].tolist())
print('mean_reward:', [round(float(x),3) for x in d['results'].mean(axis=1)])
print('mean_ep_len:', [round(float(x),1) for x in d['ep_lengths'].mean(axis=1)])
