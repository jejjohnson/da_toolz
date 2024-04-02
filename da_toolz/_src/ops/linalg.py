import cola
from jaxtyping import Float, Array
from cola.linalg import CG, Cholesky, Auto


SOLVER_ALG = dict(cholesky=Cholesky, cg=CG, auto=Auto)


def solve_linear_psd(A, b, algorithm: str="auto", *args, **kwargs):
    """
    Compute A^-1 b

    Calculates
        A x = b
          x = inv(A) @ b
    
    Parameters
    ----------
    A: jax.Array
        A in term Ax=b
    b: jax.Array
        b in term Ax=b

    Returns
    -------
    float
        returns x from term Ax=b
    """
    # create PSD matrix
    A = cola.PSD(A)
    
    # initialize the algorithm
    alg = SOLVER_ALG[algorithm](*args, **kwargs)
    
    # solve the inverse problem
    return cola.solve(A=A, b=b, alg=alg)


def inverse_psd(A, algorithm: str="auto", *args, **kwargs):
    """
    Compute the inverse of a PSD matrix

    Calculates

        y = inv(A)
    """
    # create PSD matrix
    A = cola.PSD(A)
    
    # initialize the algorithm
    alg = SOLVER_ALG[algorithm](*args, **kwargs)
    
    # solve the inverse problem
    return cola.inv(A=A, alg=alg)


def calculate_quadratic_form_psd(
    A: Float[Array, "N N"], 
    x: Float[Array, "N D"],
    algorithm: str="auto",
    *args, **kwargs
) -> Float[Array, "D D"]:
    """Calculate the quadratic form.

    Calculates

        y = x.T * inv(A) * x

    using the Cholesky decomposition of A without actually computing inv(A)
    Note that A has to be symmetric and positive definite.

    Parameters
    ----------
    A: jax.Array
        A in term y = x.T * inv(A) * x
    x: jax.Array
        x in term y = x.T * inv(A) * x

    Returns
    -------
    float
        returns the result of x.T * inv(A) * x
    """
    # calculate the inverse
    A_inv = inverse_psd(A=A, algorithm=algorithm, *args, **kwargs)
    # calculate the quadratic form
    return x.T @ A_inv @ x


def log_det_psd(A, log_algorithm: str="auto", *args, **kwargs):
    """
    Compute the log determinant of A
    
    Calculates

            y = log |A|    

    Parameters
    ----------
    A: jax.Array
        A in term y = log |A|

    Returns
    -------
    float
        returns the result of log |A|
    """
    # create PSD matrix
    A = cola.PSD(A)
    
    # initialize the algorithm
    alg = SOLVER_ALG[log_algorithm](*args, **kwargs)
    
    # solve the inverse problem
    return cola.logdet(A=A)


def create_psd_matrix(A, diagonal_boost: float=1e-9):
    # make it symmetric + boost
    return cola.PSD(make_symmetric(A) + diagonal_boost * cola.ops.I_like(A))


def make_symmetric(A):
    return 0.5 * (A + A.T)