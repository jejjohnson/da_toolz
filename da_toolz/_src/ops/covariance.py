import numpy as np


def get_dist(i, j, K):
    """Compute the absolute distance between two element indices
    within a square matrix of size (K x K)

    Args:
        i: the ith row index
        j: the jth column index
        K: shape of square array

    Returns:
        Distance
    """
    return abs(i - j) if abs(i - j) <= K / 2 else K - abs(i - j)


def gaspari_cohn(distance, radius):
    """Compute the appropriate distance dependent weighting of a
    covariance matrix, after Gaspari & Cohn, 1999 (https://doi.org/10.1002/qj.49712555417)

    Args:
        distance: the distance between array elements
        radius: localization radius for DA

    Returns:
        distance dependent weight of the (i,j) index of a covariance matrix
    """
    if distance == 0:
        weight = 1.0
    else:
        if radius == 0:
            weight = 0.0
        else:
            ratio = distance / radius
            weight = 0.0
            if ratio <= 1:
                weight = (
                    -(ratio**5) / 4
                    + ratio**4 / 2
                    + 5 * ratio**3 / 8
                    - 5 * ratio**2 / 3
                    + 1
                )
            elif ratio <= 2:
                weight = (
                    ratio**5 / 12
                    - ratio**4 / 2
                    + 5 * ratio**3 / 8
                    + 5 * ratio**2 / 3
                    - 5 * ratio
                    + 4
                    - 2 / 3 / ratio
                )
    return weight


def localize_covariance(B, loc=0):
    """Localize the model climatology covariance matrix, based on
    the Gaspari-Cohn function.

    Args:
        B: Covariance matrix over a long model run 'M_truth' (K, K)
        loc: spatial localization radius for DA

    Returns:
        Covariance matrix scaled to zero outside distance 'loc' from diagonal and
        the matrix of weights which are used to scale covariance matrix
    """
    M, N = B.shape
    X, Y = np.ix_(np.arange(M), np.arange(N))
    
    dist = np.vectorize(get_dist)(X, Y, M)
    
    W = np.vectorize(gaspari_cohn)(dist, loc)
    return B * W, W