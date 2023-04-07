import matplotlib.pyplot as plt
import numpy as np

# Sample data
labels = ['A', 'B', 'C']
values = [10, 20, 15]

# Create a bar chart
fig, ax = plt.subplots()
rects = ax.bar(labels, values)
ax.bar_label(rects, padding=3)

# # Add values on top of the bars
# for i, v in enumerate(values):
#     ax.text(i, v, str(v), ha='center', va='bottom')

# Set the x and y axis labels
ax.set_xlabel('Categories')
ax.set_ylabel('Values')

# Show the plot
plt.show()
