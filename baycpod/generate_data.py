from __future__ import division

import numpy as np

def generate_normal_time_series(num, minl=50, maxl=1000, seed=100):
    """
    Generate a series of time series data from normal distributions.

    Each time series is generated with random length, mean, and variance. The lengths
    of the time series are uniformly distributed between `minl` and `maxl`. The mean
    and variance for each segment are drawn from normal distributions, with the variance
    being adjusted to ensure positivity.

    Parameters
    ----------
    num : int
        The number of time series segments to generate.
    minl : int, optional
        The minimum length of any time series segment. Default is 50.
    maxl : int, optional
        The maximum length of any time series segment. Default is 1000.
    seed : int, optional
        The seed for the random number generator. Default is 100.

    Returns
    -------
    partition : numpy.ndarray
        An array of integers representing the lengths of the generated time series segments.
    data : numpy.ndarray
        A 2D array where each row represents a data point in the generated time series. The time series
        segments are concatenated to form a single time series.

    Notes
    -----
    The function sets the random seed at the beginning to ensure reproducibility.
    """
    np.random.seed(seed)

    # Pre-allocate data array with the total length for efficiency
    total_length = np.random.randint(minl, maxl, num).sum()
    data = np.empty(total_length, dtype=np.float64)
    
    partition = np.random.randint(minl, maxl, num)
    start_idx = 0
    for p in partition:
        mean = np.random.randn() * 10
        # Ensure variance is positive
        var = abs(np.random.randn() * 1)
        tdata = np.random.normal(mean, var, p)
        
        end_idx = start_idx + p
        data[start_idx:end_idx] = tdata
        start_idx = end_idx

    return partition, np.atleast_2d(data).T

def generate_multinormal_time_series(num, dim, minl=50, maxl=1000, seed=100):
    """
    Generate a series of time series data from multivariate normal distributions.

    Each time series segment is generated with a random length, mean vector, and covariance matrix.
    The lengths of the time series segments are uniformly distributed between `minl` and `maxl`.
    The mean vectors and covariance matrices for each segment are drawn from standard normal
    distributions, with covariance matrices being symmetric positive definite.

    Parameters
    ----------
    num : int
        The number of time series segments to generate.
    dim : int
        The dimensionality of the multivariate normal distribution.
    minl : int, optional
        The minimum length of any time series segment. Default is 50.
    maxl : int, optional
        The maximum length of any time series segment. Default is 1000.
    seed : int, optional
        The seed for the random number generator. Default is 100.

    Returns
    -------
    partition : numpy.ndarray
        An array of integers representing the lengths of the generated time series segments.
    data : numpy.ndarray
        A 2D array where each row represents a multivariate data point in the generated time series.
        The time series segments are concatenated to form a single time series.

    Notes
    -----
    The function sets the random seed at the beginning to ensure reproducibility. The covariance matrices
    are generated to be symmetric and positive definite by constructing them as A*A.T, where A is a
    matrix with entries drawn from a standard normal distribution.
    """
    np.random.seed(seed)
    
    # Pre-compute the total length for efficiency
    total_length = np.sum(np.random.randint(minl, maxl, num))
    data = np.empty((total_length, dim), dtype=np.float64)
    
    partition = np.random.randint(minl, maxl, num)
    start_idx = 0
    for p in partition:
        mean = np.random.standard_normal(dim) * 10
        # Generate a symmetric positive definite covariance matrix
        A = np.random.standard_normal((dim, dim))
        cov = np.dot(A, A.T)

        tdata = np.random.multivariate_normal(mean, cov, p)
        
        end_idx = start_idx + p
        data[start_idx:end_idx, :] = tdata
        start_idx = end_idx

    return partition, data

def generate_xuan_motivating_example(minl=50, maxl=1000, seed=100):
    """
    Generate time series data from multivariate normal distributions with predefined covariance matrices.

    This function creates time series segments from three different 2D multivariate normal distributions,
    each with a specified covariance matrix. The length of each segment is randomly determined.

    Parameters
    ----------
    minl : int, optional
        The minimum length of any time series segment. Default is 50.
    maxl : int, optional
        The maximum length of any time series segment. Default is 1000.
    seed : int, optional
        The seed for the random number generator. Default is 100.

    Returns
    -------
    partition : numpy.ndarray
        An array of integers representing the lengths of the generated time series segments.
    data : numpy.ndarray
        A 2D array where each row represents a multivariate data point in the generated time series.
        The segments are concatenated to form a single time series.

    Notes
    -----
    The covariance matrices are predefined to demonstrate the effect of changing correlations
    within the data on changepoint detection algorithms. This example is motivated by the work of Xuan et al.
    """
    np.random.seed(seed)
    dim = 2  # Dimensionality of the multivariate normal distribution
    num = 3  # Number of segments/time series to generate

    # Randomly determine the length of each segment
    partition = np.random.randint(minl, maxl, num)

    # Define mean vector and covariance matrices for each segment
    mu = np.zeros(dim)
    cov_matrices = [
        np.asarray([[1.0, 0.75], [0.75, 1.0]]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[1.0, -0.75], [-0.75, 1.0]])
    ]

    # Pre-allocate data array for efficiency
    total_length = partition.sum()
    data = np.empty((total_length, dim), dtype=np.float64)

    start_idx = 0
    for i, p in enumerate(partition):
        # Generate segment data with predefined covariance matrix
        segment_data = np.random.multivariate_normal(mu, cov_matrices[i], p)
        
        end_idx = start_idx + p
        data[start_idx:end_idx, :] = segment_data
        start_idx = end_idx

    return partition, data

