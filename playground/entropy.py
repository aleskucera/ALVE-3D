import torch

prediction = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                           [0.2, 0.3, 0.4, 0.1],
                           [0.3, 0.3, 0.2, 0.2],
                           [0.5, 0.2, 0.29, 0.01]], dtype=torch.float32)

print(f'Prediction: {prediction.shape}')

# Calculate the entropy
eps = 1e-6
prob = torch.clamp(prediction, eps, 1.0 - eps)
h = torch.sum(- prob * torch.log(prob), dim=1)

# Calculate the mean
mean = torch.mean(h)

print(f'Entropy: {h}')
print(f'Mean: {mean}')
