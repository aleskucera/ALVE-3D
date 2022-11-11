import numpy as np


#
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import matplotlib as mpl
#
# norm = mpl.colors.Normalize(vmin=0, vmax=1)
# cmap = cm.jet
# m = cm.ScalarMappable(norm=norm, cmap=cmap)
#
# # generate random data
# x = np.random.randn(1000)
#
# # plot histogram
# n, bins, patches = plt.hist(x, 50, density=True, facecolor='g', alpha=0.75)
#
# for thisfrac, thispatch in zip(n, patches):
#     color = m.to_rgba(thisfrac)
#     thispatch.set_facecolor(color)
#
# # show histogram
# plt.show()

# iterator which with iteration adds 1 to the value
class Counter:
    def __init__(self, start=0):
        self.value = start

    def __iter__(self):
        return self

    def __next__(self):
        self.value += 1
        return self.value

    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)


c = Counter()
print(c)
print(next(c))
print(next(c))
print(next(c))
