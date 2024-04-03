from typing import NamedTuple,Optional
import einx
from jaxtyping import Float, Array
import numpy as np
from sklearn.utils.extmath import randomized_svd, _randomized_eigsh
from scipy.ndimage import laplace


def pod_temporal_reconstruction(
    X: np.ndarray, 
    n_components: int=10, 
    random_state: int=42,
    localization: Optional[np.ndarray]=None
) -> np.ndarray:

    n_space, n_time = X.shape

    # calculate covariance matrix
    C_t: Float[Array, "Dt Dt"] = (X.T @ X) / (n_time - 1)

    if localization is not None:
        C_t *= localization
        # C_t = laplace(C_t)

    # calculate EVD
    S, PHI_s = _randomized_eigsh(C_t, n_components=n_components, random_state=random_state)

    # calculate time coefficients
    A_t = einx.dot("Ds Dt, Dt D -> Ds D", X, PHI_s)

    # calculate reconstruction
    X_recon = einx.dot("Ds D, Dt D -> Ds Dt", A_t, PHI_s)

    return X_recon


def pod_spatial_reconstruction(
    X: np.ndarray, 
    n_components: int=10, 
    random_state: int=42,
    localization: Optional[np.ndarray]=None
) -> np.ndarray:

    n_space, n_time = X.shape

    # calculate correlation matrix
    C_s: Float[Array, "Ds Ds"] = (X @ X.T) / (n_time - 1)

    if localization is not None:
        C_s *= localization

    # calculate EVD
    S_s, A_s = _randomized_eigsh(C_s, n_components=n_components, random_state=random_state)

    # spatial coefficients
    PHI_s: Float[Array, "Dt d"] = einx.dot("Ds Dt, Ds d -> Dt d", X, A_s)

    # reconstruction
    X_recon: Float[Array, "Ds Dt"] = einx.dot("Ds d, Dt d -> Ds Dt", A_s, PHI_s)

    return X_recon


def pod_svd_reconstruction(
    X: np.ndarray, 
    n_components: int=10, 
    random_state: int=42,
) -> np.ndarray:

    # calculate SVD
    U, S, VT = randomized_svd(X, n_components=n_components, random_state=random_state)

    # calculate reconstruction
    X_recon = U @ np.diag(S) @ VT

    return X_recon



def expectation_step(X: Float[Array, "N D"], mask: Float[Array, "N D"]) -> Float[Array, "N d"]:

    
    return None