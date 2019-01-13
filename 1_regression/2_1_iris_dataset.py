from sklearn.datasets import load_iris
iris = load_iris()

iris.keys()
print(iris.keys())

n_samples, n_features = iris.data.shape
print((n_samples, n_features))
print(iris.data[0])

print(iris.data.shape)
print(iris.target.shape)	

print(iris.target)
print(iris.target_names)

import numpy as np
import matplotlib.pyplot as plt

x_index = 0
y_index = 1

# This formatter will label the colorbar with the correct target names.
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

plt.scatter(iris.data[:, x_index], iris.data[:, y_index],
            c=iris.target, cmap=plt.cm.get_cmap('RdYlBu', 3))
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.clim(-0.5, 2.5)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index]);

plt.show()