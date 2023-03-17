from mpl_toolkits import mplot3d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from src.losses import LovaszSoftmax

matplotlib.use('TkAgg')

fig = plt.figure()

loss_fn = LovaszSoftmax()

# prediction = torch.tensor([[0.2, 0.8, 0.2],
#                            [0.8, 0.2, 0.8]]).unsqueeze(0).unsqueeze(-1)
# target = torch.tensor([1, 0, 1]).unsqueeze(0).unsqueeze(-1)
#
# loss = loss_fn(prediction, target)

x = torch.linspace(-5, 5, 30)
y = torch.linspace(-5, 5, 30)

X, Y = torch.meshgrid(x, y)
Z = loss_fn(X, Y)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('surface')

plt.show()
