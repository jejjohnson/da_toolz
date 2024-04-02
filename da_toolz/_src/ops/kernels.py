from typing import Callable
import jax
from jaxtyping import Float, Array
import jax.numpy as jnp



def sqeuclidean_distance(x: Float[Array, "D"], y: Float[Array, "D"]) -> Float[Array, ""]:
    return jnp.sum((x - y) ** 2)

def kernel_rbf(x: Float[Array, "D"], y: Float[Array, "D"], length_scale=1.0, variance=1.0) -> Float[Array, ""]:
    return variance * jnp.exp(- sqeuclidean_distance(x/length_scale, y/length_scale))

def gram(kernel_fn: Callable, x: Float[Array, "Nx D"], y: Float[Array, "Ny D"], *args, **kwargs) -> Float[Array, "Nx Ny"]:
    return jax.vmap(lambda x1: jax.vmap(lambda y1: kernel_fn(x1, y1, *args, **kwargs))(y))(x)