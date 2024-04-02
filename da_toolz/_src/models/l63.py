from typing import Tuple
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array


class L63Params(eqx.Module):
    sigma: Array
    beta: Array
    rho: Array
    
    def __init__(self, sigma: float=10.0, beta: float=8.0/3.0, rho: float=28.0):
        self.sigma = jnp.asarray(sigma)
        self.beta = jnp.asarray(beta)
        self.rho = jnp.asarray(rho)
        
        
class L63State(eqx.Module):
    x: Array
    y: Array
    z: Array
    def __init__(self, x: Array, y: Array, z: Array):
        self.x = jnp.asarray(x)
        self.y = jnp.asarray(y)
        self.z = jnp.asarray(z)
        
    def to_array(self):
        return jnp.stack([self.x, self.y, self.z],axis=1)
    
    @classmethod
    def from_array(cls, array: Array):
        return cls(x=array[..., 0], y=array[..., 1], z=array[...,2]) 
    
    
def l63_equation_of_motion(t, state: L63State, params: L63Params) -> L63State:

    # calculate derivatives
    dx, dy, dz = rhs_lorenz_63(
        x=state.x, y=state.y, z=state.z,
        sigma=params.sigma, beta=params.beta, rho=params.rho
    )

    # create new state
    state = L63State(x=dx, y=dy, z=dz)
    return state

def rhs_lorenz_63(
    x: Array,
    y: Array,
    z: Array,
    sigma: Array = 10,
    rho: Array = 28,
    beta: Array = 2.667,
) -> Tuple[Array, Array, Array]:
    x_dot = sigma * (y - x)
    # y_dot = rho * x - y - x * z
    y_dot = x * (rho - z) - y
    # x[0]*(rho-x[2])-x[1]
    z_dot = x * y - beta * z

    return x_dot, y_dot, z_dot