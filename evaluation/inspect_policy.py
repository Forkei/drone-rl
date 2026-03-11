import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from envs.nav_aviary import NavAviary

model = PPO.load('./models_nav/best_model', env=Monitor(NavAviary(target_range=2.0)), device='cpu')
p = model.policy

print('=== Policy structure ===')
print('mlp_extractor.policy_net:', p.mlp_extractor.policy_net)
print('mlp_extractor.value_net:', p.mlp_extractor.value_net)
print('action_net:', p.action_net)
print('value_net:', p.value_net)
print('log_std shape:', p.log_std.shape)
print('log_std value:', p.log_std.data)
print('policy std (exp):', th.exp(p.log_std).data)
print()
print('=== Optimizer ===')
print(type(p.optimizer))
print('param groups:', len(p.optimizer.param_groups))
print('lr:', p.optimizer.param_groups[0]['lr'])
print()
print('=== All named params requiring grad ===')
for name, param in p.named_parameters():
    print(f'  {name}: {param.shape}, requires_grad={param.requires_grad}')
