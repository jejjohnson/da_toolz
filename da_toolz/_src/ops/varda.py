from jaxtyping import Float, Array
import cola
from cola.linalg import Auto
from .linalg import create_psd_matrix

LINOP = cola.ops.operators.LinearOperator


def linear_3dvar_model_space(
    y: Float[Array, "Dy"], 
    H: Float[Array, "Dy Dx"], 
    Sigma_y: LINOP,
    mu_x: Float[Array, "Dx"],
    Sigma_x: LINOP,
    inv_alg=Auto(),
    solve_alg=Auto(),
):
    """
    Oparation:
     u = (Σu + Hᵀ Σy⁻¹ H)⁻¹(Σu⁻¹ μᵤ + Hᵀ Σy⁻¹ y)
    """
    
    # calculate the inversion
    Sigma_x_inv = cola.inv(Sigma_x, alg=inv_alg)
    Sigma_y_inv = cola.inv(Sigma_y, alg=inv_alg)
    
    # calculate matrices
    A = Sigma_x_inv + H.T @ Sigma_y_inv @ H
    b = Sigma_x_inv @ mu_x + H.T @ Sigma_y_inv @ y
    
    # make it PSD
    A = create_psd_matrix(A)
    
    # solve linear system
    ua = cola.solve(A, b, alg=solve_alg)
    return ua


def linear_3dvar_model_space_incremental(
    y: Float[Array, "Dy"], 
    H: Float[Array, "Dy Dx"], 
    Sigma_y: LINOP,
    mu_x: Float[Array, "Dx"],
    Sigma_x: LINOP,
    inv_alg=Auto(),
    solve_alg=Auto(),
):
    """
    Oparation:
     u = μᵤ + (Σu + Hᵀ Σy⁻¹ H)⁻¹ HᵀΣy⁻¹ (y - Hμᵤ)
    """
    
    # calculate the inversion
    Sigma_x_inv = cola.inv(Sigma_x, alg=inv_alg)
    Sigma_y_inv = cola.inv(Sigma_y, alg=inv_alg)
    
    # calculate matrice
    A = Sigma_x_inv + H.T @ Sigma_y_inv @ H
    innovation = y - H @ mu_x
    b = H.T @ Sigma_y_inv @ innovation
    
    # make it PSD
    A = create_psd_matrix(A)
    
    
    # solve linear system
    ua = mu_x + cola.solve(A, b, alg=solve_alg)
    return ua


def linear_3dvar_obs_space_incremental(
    y: Float[Array, "Dy"], 
    H: Float[Array, "Dy Dx"], 
    Sigma_y: LINOP,
    mu_u: Float[Array, "Dx"],
    Sigma_u: LINOP,
    solve_alg=Auto(),
):
    """
    Oparation:
     u = μᵤ + Σu Hᵀ(Σy + H Σu Hᵀ)⁻¹(y - Hμᵤ)
    """
    
    # calculate the inversion terms
    A = Sigma_y + H @ Sigma_u @ H.T
    A = create_psd_matrix(A)
    b = y - H @ mu_u
    
    # solve linear system
    ua = mu_u + Sigma_u @ H.T @ cola.solve(A, b, alg=solve_alg)
    return ua