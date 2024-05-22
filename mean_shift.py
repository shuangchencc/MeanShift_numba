import numpy as np
from numba import jit, prange
from numba.typed import List


@jit(nopython=True, cache=True, fastmath=True)
def _euclidean_dist(pointA, pointB):
    return np.sqrt(np.mean(np.power(pointA - pointB, 2)))

@jit(nopython=True, cache=True, fastmath=True)
def _distance_to_group(point, group):
    min_distance = np.finfo(np.float32).max
    for idx in range(len(group)):
        pt = group[idx]
        dist = _euclidean_dist(point, pt)
        if dist < min_distance:
            min_distance = dist
    return min_distance
    
@jit(nopython=True, cache=True, fastmath=True)
def _determine_nearest_group(point, groups):    
    GROUP_DISTANCE_TOLERANCE = .1
    nearest_group_index = -9999
    index = 0
    for idx in range(len(groups)):
        group = groups[idx]
        distance_to_group = _distance_to_group(point, group)
        if distance_to_group < GROUP_DISTANCE_TOLERANCE:
            nearest_group_index = index
        index += 1
    return nearest_group_index
    
@jit(nopython=True, cache=True, fastmath=True)
def _group_points(points):
    group_assignment = np.array([0,])
    groups = List( points[0:1] )
    group_index = 1
    for idx in range(1, len(points)):
        point = points[idx]
        nearest_group_index = _determine_nearest_group(point, groups)
        if nearest_group_index == -9999:
            # create new group
            group_assignment = np.append(group_assignment, group_index)
            group_index += 1
            groups.append( point )
        else:
            group_assignment = np.append(group_assignment, nearest_group_index)
            # print(type(groups[nearest_group_index]), groups[nearest_group_index].shape)
            groups[nearest_group_index] = np.append(groups[nearest_group_index], points[idx:idx+1])
    return group_assignment

@jit(nopython=True, cache=True, fastmath=True)
def _multivariate_gaussian_kernel(distances, bandwidths):
    # Number of dimensions of the multivariate gaussian
    dim = len(bandwidths)
    # Covariance matrix
    cov = np.multiply(np.power(bandwidths, 2), np.eye(dim))
    # Compute Multivariate gaussian (vectorized implementation)
    exponent = -0.5 * np.sum(np.multiply(np.dot(distances, np.linalg.inv(cov)), distances), axis=1)
    val = (1 / np.power((2 * np.pi), (dim/2)) * np.power(np.linalg.det(cov), 0.5)) * np.exp(exponent)
    return val
    
@jit(nopython=True, cache=True, fastmath=True)
def _shift_point(point, points, kernel_bandwidth):
    # from http://en.wikipedia.org/wiki/Mean-shift
    # points = np.array(points)
    # numerator
    point_weights = _multivariate_gaussian_kernel(point-points, kernel_bandwidth)
    # tiled_weights = np.tile(point_weights, [len(point), 1])
    tiled_weights = np.repeat(point_weights, len(point)).reshape((len(point_weights), len(point))).T
    # denominator
    denominator = sum(point_weights)
    shifted_point = np.multiply(tiled_weights.transpose(), points).sum(axis=0) / denominator
    return shifted_point

def mean_shift(points, kernel_bandwidth):
    MIN_DISTANCE = 0.000001
    
    shift_points = points.copy()
    max_min_dist = 1
    iteration_number = 0

    still_shifting = np.ones((points.shape[0],), dtype=np.bool_)
    while max_min_dist > MIN_DISTANCE:
        # print max_min_dist
        max_min_dist = 0
        iteration_number += 1
        for i in range(0, len(shift_points)):
            if not still_shifting[i]:
                continue
            p_new = shift_points[i]
            p_new_start = p_new
            p_new = _shift_point(p_new, points, kernel_bandwidth)
            dist = _euclidean_dist(p_new, p_new_start)
            if dist > max_min_dist:
                max_min_dist = dist
            if dist < MIN_DISTANCE:
                still_shifting[i] = False
            shift_points[i] = p_new

    group_assignments = _group_points(shift_points)
    return group_assignments
