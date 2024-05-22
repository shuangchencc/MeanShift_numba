# Mean Shift Clustering in numba
A fast implementation of mean shift clustering in numba, python

## Dependencies
numpy and numba

## Description
The mean_shift.py module provides a function called `mean_shift`. 
This function takes two main input variables, `points` and `kernel_bandwidth`. 
The `points` is a numpy array of shape [num_points, dimension], and the `kernel_bandwidth` is also a numpy array of shape [dimension,].
The output of `mean_shift` is the assigned labels as a numpy array of shape [num_points,].

## Usage
```
import numpy as np
import matplotlib.pyplot as plt

from mean_shift import mean_shift

## 1. one-dimensional clustering
points = np.random.rand(100, 1)
points[:20]   += 2
points[20:40] += 4
points[40:60] += 5

clusters = mean_shift(points, np.array((0.5, )), GROUP_DISTANCE_TOLERANCE=1e-1)
for idx in set(clusters):
    plt.scatter(range(len(points[clusters==idx, 0])), points[clusters==idx, 0])
    
## 2. two-dimensional clustering
points = np.random.rand(100, 2)
points[:20]   += (0,2)
points[20:40] += (2,0)
points[40:60] += (2,2)

clusters = mean_shift(points, np.array((0.5, 0.5)), GROUP_DISTANCE_TOLERANCE=1e-1)
plt.figure()
for idx in set(clusters):
    plt.scatter(points[clusters==idx, 0], points[clusters==idx, 1])
```

## Reference
The code is based on this repo [MeanShift_py](https://github.com/mattnedrich/MeanShift_py).
