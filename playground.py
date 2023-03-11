import torch

tensor_1 = torch.tensor([1, 2, 3, 4, 5])
mask = torch.tensor([True, False, True, False, True])

tensor_2 = tensor_1[mask]

print(tensor_2)
