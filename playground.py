import torch

# Test the gradient

x = torch.randn(1, 3, 4, 4, requires_grad=True)
y = x * 2
y = y.squeeze(0).flatten(start_dim=1).permute(1, 0)

# Show the gradient
print(x.grad)
