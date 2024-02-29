from abc import ABC, abstractmethod
from decorator import decorator

import numpy as np
import torch
import scipy.stats as ss
from scipy.special import gammaln, multigammaln, comb


# def _dynamic_programming(f, *args, **kwargs):
#     if f.data is None:
#         f.data = args[1]

#     if not np.array_equal(f.data, args[1]):
#         f.cache = {}
#         f.data = args[1]

#     try:
#         f.cache[args[2:4]]
#     except KeyError:
#         f.cache[args[2:4]] = f(*args, **kwargs)
#     return f.cache[args[2:4]]

def _dynamic_programming(f, *args, **kwargs):
    # Ensure data is properly initialized and compared. This part might need adjustment based on actual use.
    if f.data is None:
        f.data = args[1]

    # Convert numpy arrays to a hashable state for comparison (e.g., using a tuple of tuples for 2D arrays)
    if isinstance(args[1], np.ndarray):
        data_key = tuple(map(tuple, args[1]))
    else:
        data_key = args[1]

    if not np.array_equal(f.data, args[1]):
        f.cache = {}
        f.data = args[1]

    # Convert args (excluding 'data' which is args[1]) to a fully hashable state for caching
    cache_args = tuple([arg if isinstance(arg, (int, float, str, tuple)) else tuple(arg) for arg in args[2:4]])

    # Construct a cache key that includes both the arguments and the data key
    cache_key = (cache_args, data_key)

    try:
        return f.cache[cache_key]
    except KeyError:
        f.cache[cache_key] = f(*args, **kwargs)
    return f.cache[cache_key]


def dynamic_programming(f):
    f.cache = {}
    f.data = None
    return decorator(_dynamic_programming, f)





class BaseLikelihood(ABC):
    """
    This is an abstract class to serve as a template for future users to mimick
    if they want to add new models for offline bayesian changepoint detection.

    Make sure to override the abstract methods to do which is desired.
    Otherwise you will get an error.
    """

    @abstractmethod
    def pdf(self, data: np.array, t: int, s: int):
        raise NotImplementedError(
            "PDF is not defined. Please define in separate class and override this function."
        )


class IndepentFeaturesLikelihood:
    """
    Return the pdf for an independent features model discussed in xuan et al

    Parmeters:
        data - the datapoints to be evaluated (shape: 1 x D vector)
        t - start of data segment
        s - end of data segment
    """

    def pdf(self, data: np.array, t: int, s: int):
        s += 1
        n = s - t
        x = data[t:s]
        if len(x.shape) == 2:
            d = x.shape[1]
        else:
            d = 1
            x = np.atleast_2d(x).T

        N0 = d  # weakest prior we can use to retain proper prior
        V0 = np.var(x)
        Vn = V0 + (x ** 2).sum(0)

        # sum over dimension and return (section 3.1 from Xuan paper):
        return d * (
            -(n / 2) * np.log(np.pi)
            + (N0 / 2) * np.log(V0)
            - gammaln(N0 / 2)
            + gammaln((N0 + n) / 2)
        ) - (((N0 + n) / 2) * np.log(Vn)).sum(0)


class FullCovarianceLikelihood:
    def pdf(self, data: np.ndarray, t: int, s: int):
        """
        Return the pdf function for the covariance model discussed in xuan et al

        Parmeters:
            data - the datapoints to be evaluated (shape: 1 x D vector)
            t - start of data segment
            s - end of data segment
        """
        s += 1
        n = s - t
        x = data[t:s]
        if len(x.shape) == 2:
            dim = x.shape[1]
        else:
            dim = 1
            x = np.atleast_2d(x).T

        N0 = dim  # weakest prior we can use to retain proper prior
        V0 = np.var(x) * np.eye(dim)

        # Improvement over np.outer
        # http://stackoverflow.com/questions/17437523/python-fast-way-to-sum-outer-products
        # Vn = V0 + np.array([np.outer(x[i], x[i].T) for i in xrange(x.shape[0])]).sum(0)
        Vn = V0 + np.einsum("ij,ik->jk", x, x)

        # section 3.2 from Xuan paper:
        return (
            -(dim * n / 2) * np.log(np.pi)
            + (N0 / 2) * np.linalg.slogdet(V0)[1]
            - multigammaln(N0 / 2, dim)
            + multigammaln((N0 + n) / 2, dim)
            - ((N0 + n) / 2) * np.linalg.slogdet(Vn)[1]
        )


class StudentT(BaseLikelihood):
    @dynamic_programming
    def pdf(self, data: np.ndarray, t: int, s: int):
        """
        Return the pdf function of the t distribution
        Uses update approach in https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf (page 8, 89)

        Parmeters:
            data - the datapoints to be evaluated (shape: 1 x D vector)
            t - start of data segment
            s - end of data segment
        """
        s += 1
        n = s - t

        mean = data[t:s].sum(0) / n
        muT = (n * mean) / (1 + n)
        nuT = 1 + n
        alphaT = 1 + n / 2

        betaT = (
            1
            + 0.5 * ((data[t:s] - mean) ** 2).sum(0)
            + ((n) / (1 + n)) * (mean ** 2 / 2)
        )
        scale = (betaT * (nuT + 1)) / (alphaT * nuT)

        # splitting the PDF of the student distribution up is /much/ faster.
        # (~ factor 20) using sum over for loop is even more worthwhile
        # prob = np.sum(np.log(1 + (data[t:s] - muT) ** 2 / (nuT * scale)))
        prob = adaptive_log_sum(data, t, s, muT, nuT, scale)
        
        # lgA = (
        #     gammaln((nuT + 1) / 2)
        #     - np.log(np.sqrt(np.pi * nuT * scale))
        #     - gammaln(nuT / 2)
        # )
        # result = lgA + (nuT + 1) / 2 * np.log(1 + (data[t:s] - muT) ** 2 / (nuT * scale))

        result = compute_log_gaussian(data, t, s, muT, nuT, scale, device=None)

        return result

def compute_log_gaussian(data, t, s, muT, nuT, scale, device=None):
    """
    Computes a log Gaussian computation that is compatible with both NumPy arrays and PyTorch tensors.

    Parameters:
    - data (numpy.ndarray or torch.Tensor): The input data.
    - t (int): Start index.
    - s (int): End index.
    - muT (float or tensor): Mean parameter.
    - nuT (float or tensor): Degrees of freedom parameter.
    - scale (float or tensor): Scale parameter.
    - device (str, optional): Specifies the device for PyTorch computations. Defaults to None.

    Returns:
    - The computed result as a NumPy array or PyTorch tensor.
    """
    if isinstance(data, torch.Tensor):
        # Ensure parameters are tensors and on the correct device
        if device is None:
            device = data.device
        muT = torch.tensor(muT, device=device, dtype=torch.float32)
        nuT = torch.tensor(nuT, device=device, dtype=torch.float32)
        scale = torch.tensor(scale, device=device, dtype=torch.float32)
        
        data_segment = data[t:s] - muT
        lgA = (torch.lgamma((nuT + 1) / 2) -
               torch.log(torch.sqrt(torch.tensor(np.pi, device=device) * nuT * scale)) -
               torch.lgamma(nuT / 2))
        result = lgA + (nuT + 1) / 2 * torch.log(1 + torch.pow(data_segment, 2) / (nuT * scale))
    else:
        # Use NumPy for computation
        data_segment = data[t:s] - muT
        lgA = (gammaln((nuT + 1) / 2) -
               np.log(np.sqrt(np.pi * nuT * scale)) -
               gammaln(nuT / 2))
        result = lgA + (nuT + 1) / 2 * np.log(1 + np.power(data_segment, 2) / (nuT * scale))

    return result



def adaptive_log_sum(data, start, end, muT, nuT, scale):
    """
    Computes the log of 1 plus a squared difference, followed by a sum,
    in an adaptive manner to support both NumPy arrays and PyTorch tensors.

    Parameters:
    data (numpy.ndarray or torch.Tensor): The input data.
    start (int): The start index.
    end (int): The end index.
    muT, nuT, scale: Parameters used in the computation, must match the type of `data`.

    Returns:
    The result of the computation, in the same type as the input data.
    """
    if isinstance(data, np.ndarray):
        # Use NumPy operations
        result = np.sum(np.log(1 + (data[start:end] - muT) ** 2 / (nuT * scale)))
    elif isinstance(data, torch.Tensor):
        # Use PyTorch operations
        result = torch.sum(torch.log(1 + (data[start:end] - muT) ** 2 / (nuT * scale)))
        result = result.cpu().numpy()
    else:
        raise TypeError("Unsupported data type. Data must be a NumPy array or a PyTorch tensor.")

    return result

