import einx
import jax
import jax.numpy as jnp
import cola
from cola.linalg import Auto
from jaxtyping import Float, Array

def analysis_etkf(
    Xf: Float[Array, "Dx Ne"],
    HXf: Float[Array, "Dy Ne"],
    Y: Float[Array, "Dy"], 
    R: Float[Array, "Dy Dy"], 
    inv_alg=Auto(), 
    eigs_alg=Auto()
):
    """
    Implementation adapted from pseudocode description in
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),
    algorithm 7, see section 5.4.
    Errors: Calculation of W1 prime, divide by square root of eigenvalues. The mathematical formula in the paper has an error already.
    
    Dimensions: N_e: ensemble size, N_y: Number of observations: N_x: State vector size (Gridboxes x assimilated variables)
    
    Input:
    - Xf:  the prior ensemble ($N_x$ x $N_e$) 
    - R: Measurement Error (Variance of pseudoproxy timerseries) ($N_y$ x 1$) -> converted to Ny x Ny matrix
    - HX^f: Model value projected into observation space/at proxy locations ($N_y$ x $N_e$)
    - Y: Observation vector ($N_y$ x 1)

    Output:
    - Analysis ensemble (N_x, N_e)
    """
    # number of ensemble members
    num_ensembles = jnp.shape(Xf)[1]

    # inverse obs error matrix
    R_inv: Float[Array, "Dy Dy"] = cola.inv(R, alg=inv_alg)
    
    # Mean of prior ensemble for each gridbox   
    X_mu: Float[Array, "Dx"] = einx.mean("Dx [Ne]", Xf)
    
    # Perturbations from ensemble mean
    Xfp: Float[Array, "Dx Ne"] = einx.subtract("Dx Ne, Dx -> Dx Ne", Xf, X_mu)
    
    # Mean and perturbations for model values in observation space
    HXf_mu: Float[Array, "Dy"] = einx.mean("Dy [Ne]", HXf)
    HXp: Float[Array, "Dy Ne"] = einx.subtract("Dy Ne, Dy -> Dy Ne", HXf, HXf_mu)
    
    # do inversion
    A: Float[Array, "Ne Ne"] = cola.ops.Dense(HXp.T) @ R_inv @ cola.ops.Dense(HXp)
    A += (num_ensembles - 1) * cola.ops.I_like(A) 
    
    # eigenvalue decomposition of A2, A2 is symmetric
    eigvals, eigvecs = cola.linalg.eig(A, k=num_ensembles, alg=eigs_alg)
    
    # compute perturbations
    D: Float[Array, "Ne Ne"] = jnp.sqrt(num_ensembles - 1) * cola.ops.Diagonal(jnp.sqrt(1/eigvals))
    Wp: Float[Array, "Ne Ne"] = eigvecs @ D @ eigvecs.T

    # compute ensemble perturbation
    innovation: Float[Array, "Dy"] = Y - HXf_mu
    D: Float[Array, "Ne Ne"] = cola.ops.Diagonal(1/eigvals)
    wm: Float[Array, "Ne"] = eigvecs @ D @ eigvecs.T @ HXp.T @ R_inv @ innovation

    # adding pert and mean
    W: Float[Array, "Ne Ne"] = einx.add("Ne Nb, Ne -> Ne Nb", Wp.to_dense(), wm)
    
    # calculate perturbation correction
    X_correction: Float[Array, "Dx Ne"] = einx.dot("Dx Ne, Ne Nb -> Dx Nb", Xfp, W)
    
    # add correction
    Xa = einx.add("Dx, Dx Ne -> Dx Ne", X_mu, X_correction)

    return Xa.real