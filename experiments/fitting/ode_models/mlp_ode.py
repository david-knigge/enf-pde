import jax.numpy as jnp
import flax.linen as nn


class MLPODE(nn.Module):
    num_hidden: int
    num_layers: int
    scalar_num_out: int
    vec_num_out: int

    def setup(self):
        self.mlp_a = nn.Sequential((
            nn.Dense(self.num_hidden),
            nn.gelu,
            nn.Dense(self.num_hidden),
            nn.gelu,
            nn.Dense(self.num_hidden),
            nn.gelu,
            nn.Dense(self.scalar_num_out)
        ))
        self.mlp_p = nn.Sequential((
            nn.Dense(self.num_hidden),
            nn.gelu,
            nn.Dense(self.num_hidden),
            nn.gelu,
            nn.Dense(self.num_hidden),
            nn.gelu,
            nn.Dense(2 * self.vec_num_out)
        ))

    def __call__(self, latents):
        # Unpack latents
        p, a, window = latents

        # a is distributed with 1 mean, so subtract this
        a = a - 1

        # Get the latent derivative
        derivative_p_pos = self.mlp_p(jnp.concatenate([p, a], axis=-1))
        derivative_a = self.mlp_a(jnp.concatenate([p, a], axis=-1))

        return derivative_p_pos, derivative_a, jnp.zeros_like(window)
