import numpy as np
d = np.load('./logs_mujoco/evaluations.npz')
for k in d:
    print(k, d[k])
