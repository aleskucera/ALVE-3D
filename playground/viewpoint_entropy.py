import torch
import numpy as np

view_1 = torch.tensor([[9, 1, 3, 5]], dtype=torch.float32)
view_2 = torch.tensor([[4, 2, 3, 4]], dtype=torch.float32)
view_3 = torch.tensor([[8, 1, 4, 2]], dtype=torch.float32)
print(f'View_1: {view_1}')
print(f'View_2: {view_2}')
print(f'View_3: {view_3}')
print('')

# Create from views probability distribution
p_1 = torch.softmax(view_1, dim=1)
p_2 = torch.softmax(view_2, dim=1)
p_3 = torch.softmax(view_3, dim=1)
print(f'P_1: {p_1}')
print(f'P_2: {p_2}')
print(f'P_3: {p_3}')
print('')

# Concatenate the distributions
omega = torch.concatenate((p_1, p_2, p_3), dim=0)
print(f'Omega: {omega}')
print('')

# Calculate the mean of the distribution
mean = torch.mean(omega, dim=0)
print(f'Q: {mean}')
print('')

# Calculate the entropy of the mean
eps = 1e-6
prob = torch.clamp(mean, eps, 1.0 - eps)
h = torch.sum(- prob * torch.log(prob))
print(f'Viewpoint Entropy: {h}')
