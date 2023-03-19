import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from src.losses import LovaszSoftmax

matplotlib.use('TkAgg')

import torch
import matplotlib.pyplot as plt
import numpy as np

# Define the range of values for x and y axis
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)

# Create a grid of points from x and y values
xx, yy = np.meshgrid(x, y)

# Flatten the grid to create input tensor
inputs = torch.tensor(np.vstack([xx.ravel(), yy.ravel()]).T, dtype=torch.float32)

# Define the true labels for the inputs
labels = torch.randint(0, 2, (len(inputs),))

# Define the CrossEntropyLoss function
loss_fn = torch.nn.CrossEntropyLoss()

# Calculate the loss for each input
losses = loss_fn(torch.softmax(inputs, dim=1), labels)

# Reshape the losses to match the shape of xx and yy
losses = losses.reshape(xx.shape)

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(xx, yy, losses, cmap='coolwarm')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('CrossEntropyLoss')
plt.show()
