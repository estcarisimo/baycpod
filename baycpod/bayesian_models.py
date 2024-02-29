import numpy as np
import pymc as pm
import torch
import torch.nn.functional as F

try:
    from sselogsumexp import logsumexp
except ImportError:
    from scipy.special import logsumexp

    print("Use scipy logsumexp().")
else:
    print("Use SSE accelerated logsumexp().")


# def offline_changepoint_detection(
#     data, prior_function, log_likelihood_class, truncate: int = -40
# ):
#     """
#     Compute the likelihood of changepoints on data.

#     Parameters:
#     data    -- the time series data
#     truncate  -- the cutoff probability 10^truncate to stop computation for that changepoint log likelihood

#     Outputs:
#         P  -- the log-likelihood of a datasequence [t, s], given there is no changepoint between t and s
#         Q -- the log-likelihood of data
#         Pcp --  the log-likelihood that the i-th changepoint is at time step t. To actually get the probility of a changepoint at time step t sum the probabilities.
#     """

#     # Set up the placeholders for each parameter
#     n = len(data)
#     Q = np.zeros((n,))
#     g = np.zeros((n,))
#     G = np.zeros((n,))
#     P = np.ones((n, n)) * -np.inf

#     # save everything in log representation
#     for t in range(n):
#         g[t] = prior_function(t)
#         if t == 0:
#             G[t] = g[t]
#         else:
#             G[t] = np.logaddexp(G[t - 1], g[t])

#     P[n - 1, n - 1] = log_likelihood_class.pdf(data, t=n - 1, s=n)
#     Q[n - 1] = P[n - 1, n - 1]

#     # print(data)

#     for t in reversed(range(n - 1)):
#         P_next_cp = -np.inf  # == log(0)
#         for s in range(t, n - 1):
#             P[t, s] = log_likelihood_class.pdf(data, t=t, s=s + 1)

#             # compute recursion
#             summand = P[t, s] + Q[s + 1] + g[s + 1 - t]
#             P_next_cp = np.logaddexp(P_next_cp, summand)

#             # truncate sum to become approx. linear in time (see
#             # Fearnhead, 2006, eq. (3))
#             if summand - P_next_cp < truncate:
#                 break

#         P[t, n - 1] = log_likelihood_class.pdf(data, t=t, s=n)

#         # (1 - G) is numerical stable until G becomes numerically 1
#         if G[n - 1 - t] < -1e-15:  # exp(-1e-15) = .99999...
#             antiG = np.log(1 - np.exp(G[n - 1 - t]))
#         else:
#             # (1 - G) is approx. -log(G) for G close to 1
#             antiG = np.log(-G[n - 1 - t])

#         Q[t] = np.logaddexp(P_next_cp, P[t, n - 1] + antiG)

#     Pcp = np.ones((n - 1, n - 1)) * -np.inf
#     for t in range(n - 1):
#         Pcp[0, t] = P[0, t] + Q[t + 1] + g[t] - Q[0]
#         if np.isnan(Pcp[0, t]):
#             Pcp[0, t] = -np.inf
#     for j in range(1, n - 1):
#         for t in range(j, n - 1):
#             tmp_cond = (
#                 Pcp[j - 1, j - 1 : t]
#                 + P[j : t + 1, t]
#                 + Q[t + 1]
#                 + g[0 : t - j + 1]
#                 - Q[j : t + 1]
#             )
#             Pcp[j, t] = logsumexp(tmp_cond.astype(np.float32))
#             if np.isnan(Pcp[j, t]):
#                 Pcp[j, t] = -np.inf

#     return Q, P, Pcp

def offline_changepoint_detection(data, prior_function, log_likelihood_class, truncate = -40, method='pymc3', device='cpu'):
    """
    Offline Changepoint Detection.

    Computes the likelihood of changepoints in the data sequence using either PyMC3 or PyTorch.

    Parameters:
        data: array-like
            The time series data.
        prior_function: function
            Function to compute the prior.
        log_likelihood_class: object
            Object to compute the log likelihood.
        method: str, optional
            The method to use for computation: 'pymc3' or 'pytorch'. Defaults to 'pymc3'.
        device: str, optional
            The device to use when method is 'pytorch': 'cpu' or 'cuda'. Defaults to 'cpu'.

    Returns:
        Q: array
            Log-likelihood of a data sequence given no changepoint between t and s.
        P: array
            Log-likelihood of data.
        Pcp: array
            Log-likelihood that the i-th changepoint is at time step t. To get the probability of a changepoint at time step t, sum the probabilities.
    """
    n = len(data)
    if method == 'pymc3':
        Q, P, Pcp = offline_changepoint_detection_pymc3(data, prior_function, log_likelihood_class, truncate=truncate)
    elif method == 'pytorch':
        Q, P, Pcp = offline_changepoint_detection_pytorch(data, prior_function, log_likelihood_class, truncate=truncate, device=device)
    else:
        raise ValueError("Invalid method. Choose between 'pymc3' or 'pytorch'.")
    return Q, P, Pcp


def offline_changepoint_detection_pymc3(data, prior_function, log_likelihood_class):
    """
    Offline Changepoint Detection using PyMC3.

    Computes the likelihood of changepoints in the data sequence using PyMC3.

    Parameters:
        data: array-like
            The time series data.
        prior_function: function
            Function to compute the prior.
        log_likelihood_class: object
            Object to compute the log likelihood.

    Returns:
        Q: array
            Log-likelihood of a data sequence given no changepoint between t and s.
        P: array
            Log-likelihood of data.
        Pcp: array
            Log-likelihood that the i-th changepoint is at time step t. To get the probability of a changepoint at time step t, sum the probabilities.
    """
    n = len(data)
    Q = np.zeros(n)
    g = np.zeros(n)
    P = np.ones((n, n)) * -np.inf

    with pm.Model() as model:
        # # Assuming prior_function returns integer indices, get counts for each index
        # indices = prior_function(np.arange(n))
        # # Use bincount to count occurrences, assuming max(indices) < n for simplicity
        # counts = TT.bincount(indices, minlength=n)
        # # Use the counts as your 'g' variable or in any other way you intended
        # g = pm.Deterministic('g', counts)
        
        g = pm.Deterministic('g', prior_function(np.arange(n)))
        for t in range(n):
            P[t, t:] = log_likelihood_class.pdf(data, t=t, s=np.arange(t, n))
            Q[t] = pm.math.logsumexp(P[t, t:] + Q[t + 1:] + g[:n - t])

        # # Initialize a tensor 'a' with zeros
        # a = TT.zeros(n)
        # for i in range(n):
        #     # Call prior_function for each i to get a tensor value to assign to a[i]
        #     tensor_value = prior_function(i)
        #     # Ensure the tensor_value is a Theano tensor
        #     tensor_value = TT.as_tensor_variable(tensor_value)
        #     # Use set_subtensor to update 'a' at position 'i' with 'tensor_value'
        #     a = TT.set_subtensor(a[i], tensor_value)
        
        # # Use 'a' as the deterministic variable 'g'
        # g = pm.Deterministic('g', a)

        # for t in range(n):
        #     P[t, t:] = log_likelihood_class.pdf(data, t=t, s=np.arange(t, n))
        #     # Compute the log-likelihood efficiently
        #     Q[t] = pm.math.logsumexp(P[t, t:] + Q[t + 1:] + g[t:n])

        Pcp = np.zeros((n - 1, n - 1))
        j_idx = np.arange(1, n - 1)
        t_idx = np.arange(n - 1)
        t_grid, j_grid = np.meshgrid(t_idx, j_idx, indexing='ij')
        Pcp[j_grid, t_grid] = pm.math.logsumexp(P[j_grid, t_grid[:, None]:] + Q[t_grid[:, None] + 1] + g[:n - j_grid - 1], axis=0)

    return Q, P, Pcp


def offline_changepoint_detection_pytorch(data, prior_function, log_likelihood_class, truncate=-40, device='cpu'):
    """
    Perform offline changepoint detection on a series of data using PyTorch for computations.

    This function computes the log-likelihood of having no changepoint between two points in the series,
    the overall log-likelihood of the data, and the log-likelihood of having a changepoint at a specific time step.

    Parameters
    ----------
    data : array-like
        The time series data as a 1D array.
    prior_function : function
        A function that computes the prior probability for each time step. It should accept an integer and return a float.
    log_likelihood_class : object
        An object that has a method `pdf` to compute the log-likelihood of the data given potential changepoints. 
        The `pdf` method should accept the data, start time `t`, and end time `s`.
    truncate : int, optional
        The cutoff log probability (log base 10) used to stop computation for that changepoint log-likelihood to enhance performance. Defaults to -40.
    device : str, optional
        The device on which computations are performed ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns
    -------
    Q : numpy.ndarray
        A 1D array representing the log-likelihood of a data sequence given no changepoint between t and s.
    P : numpy.ndarray
        A 2D array representing the log-likelihood of the data.
    Pcp : numpy.ndarray
        A 2D array representing the log-likelihood that the i-th changepoint is at time step t.

    Notes
    -----
    The function relies on PyTorch for efficient computation, especially beneficial for GPU acceleration.
    """
    # Number of data points
    n = len(data)
    # Convert data to a PyTorch tensor
    data = torch.tensor(data, dtype=torch.float32, device=device)

    # Initialize tensors for storing computations
    Q = torch.zeros(n, device=device)
    P = torch.full((n, n), float('-inf'), device=device)  # Initialized to -inf for log(0)

    # Compute prior probabilities and their cumulative sum
    g = torch.tensor([prior_function(t) for t in range(n)], dtype=torch.float32, device=device)
    G = torch.cumsum(g, dim=0)

    # Compute the log-likelihood for the last data point
    Q[n-1] = log_likelihood_class.pdf(data, t=n-1, s=n).view(-1)[0].item()

    # Main loop for computing P and Q using dynamic programming
    LOWER_THRESHOLD = -1e-15  # Threshold to handle numerical stability in log calculations
    with torch.no_grad():
        for t in reversed(range(n-1)):
            P_next_cp = torch.tensor(float('-inf'), device=device)  # Initialize to -inf for log(0)
            for s in range(t, n-1):
                # Compute log-likelihood for each segment and update P and Q accordingly
                P[t, s] = log_likelihood_class.pdf(data, t=t, s=s+1).view(-1)[0].item()
                summand = P[t, s] + Q[s+1] + g[s+1-t]
                P_next_cp = torch.logaddexp(P_next_cp, summand)
                if summand - P_next_cp < truncate:  # Early stopping based on the truncate threshold
                    break

            # Compute for the last segment
            P[t, n-1] = log_likelihood_class.pdf(data, t=t, s=n).view(-1)[0].item()

            # Compute Q with consideration for numerical stability
            antiG = torch.log(1 - torch.exp(G[n-1-t])) if G[n-1-t] < LOWER_THRESHOLD else torch.log(-G[n-1-t])
            Q[t] = torch.logaddexp(P_next_cp, P[t, n-1] + antiG)

    # Compute Pcp, the probability of changepoints
    Pcp = torch.full((n - 1, n - 1), float('-inf'), device=device)
    for t in range(n - 1):
        for j in range(t, n - 1):
            # Directly compute Pcp for j == t, otherwise accumulate using logaddexp for numerical stability
            Pcp[t, j] = P[t, j] + Q[j + 1] + g[j] - Q[t] if j == t else torch.logaddexp(Pcp[t, j - 1], P[t, j] + Q[j + 1] + g[j] - Q[t])

    # It does not return the same ouput as the original function, but it is consistent with the PyTorch implementation
    # It also seems to be better in runtime performance

    # Convert results to numpy arrays before returning
    return Q.cpu().numpy(), P.cpu().numpy(), Pcp = Pcp.cpu().numpy()




def online_changepoint_detection(data, hazard_function, log_likelihood_class):
    """
    Use online bayesian changepoint detection
    https://scientya.com/bayesian-online-change-point-detection-an-intuitive-understanding-b2d2b9dc165b

    Parameters:
    data    -- the time series data

    Outputs:
        R  -- is the probability at time step t that the last sequence is already s time steps long
        maxes -- the argmax on column axis of matrix R (growth probability value) for each time step
    """
    maxes = np.zeros(len(data) + 1)

    R = np.zeros((len(data) + 1, len(data) + 1))
    R[0, 0] = 1

    for t, x in enumerate(data):
        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        predprobs = log_likelihood_class.pdf(x)

        # Evaluate the hazard function for this interval
        H = hazard_function(np.array(range(t + 1)))

        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive
        # probabilities.
        R[1 : t + 2, t + 1] = R[0 : t + 1, t] * predprobs * (1 - H)

        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0.
        R[0, t + 1] = np.sum(R[0 : t + 1, t] * predprobs * H)

        # Renormalize the run length probabilities for improved numerical
        # stability.
        R[:, t + 1] = R[:, t + 1] / np.sum(R[:, t + 1])

        # Update the parameter sets for each possible run length.
        log_likelihood_class.update_theta(x, t=t)

        maxes[t] = R[:, t].argmax()

    return R, maxes
