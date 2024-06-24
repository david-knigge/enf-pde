import jax.numpy as jnp
from flax import linen as nn


class PolynomialFeatures(nn.Module):
    degree: int

    def setup(self):
        pass  # No setup needed as there are no trainable parameters.

    def __call__(self, x):
        polynomial_list = [x]
        for it in range(1, self.degree+1):
            polynomial_list.append(jnp.einsum('...i,...j->...ij', polynomial_list[-1], x).reshape(*x.shape[:-1], -1))
        return jnp.concatenate(polynomial_list, axis=-1)


class PolynomialEmbedding(nn.Module):
    num_out: int
    num_hidden: int
    degree: int
    num_layers: int = 2

    def setup(self):
        assert (
            self.num_layers >= 2
        ), "At least two layers (the hidden plus the output one) are required."

        # Encoding
        self.encoding = PolynomialFeatures(degree=self.degree)

        # Hidden layers
        self.layers = [
            nn.Sequential((nn.Dense(self.num_hidden), nn.gelu))
            for _ in range(self.num_layers - 1)
        ]

        # Output layer
        self.linear_final = nn.Dense(
            features=self.num_out,
            use_bias=True,
        )

    def __call__(self, x):
        x = self.encoding(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = self.linear_final(x)
        return x
