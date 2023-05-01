import torch
import time

# decay_rate = 0.5
# num_values = int(1e6)
# num_labels = int(1e3)
#
# values = torch.rand(num_values)
# labels = torch.randint(0, num_labels, (num_values,))
# weights = torch.ones(num_labels)
#
# start = time.time()
# for i in range(num_values):
#     label = labels[i]
#     weight = weights[label]
#     values[i] *= weight
#     weights[label] *= decay_rate
#
# print(time.time() - start)
#
# values = torch.rand(num_values)
# labels = torch.randint(0, num_labels, (num_values,))
#
# start = time.time()
# unique_labels, counts = torch.unique(labels, return_counts=True)
# for label, count in zip(unique_labels, counts):
#     series = [decay_rate ** j for j in range(count)]
#     values[labels == label] *= torch.tensor(series)
# print(time.time() - start)

tensor1 = torch.tensor([1, 2, 3, 4, 5])
order = torch.tensor([4, 3, 2, 1, 0])
weighted_order = torch.tensor([0, 3, 2, 1, 4])

print(tensor1[order])
print(tensor1[order[weighted_order]])
