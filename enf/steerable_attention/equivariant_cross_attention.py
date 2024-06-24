import jax
import jax.numpy as jnp
from jax.nn import gelu, softmax
from flax import linen as nn

from enf.steerable_attention.embedding import get_embedding
from enf.steerable_attention.invariant._base_invariant import BaseInvariant


class PointwiseFFN(nn.Module):
    num_in: int
    num_hidden: int
    num_out: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.num_hidden)(x)
        x = gelu(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.num_out)(x)
        return x


class EquivariantCrossAttention(nn.Module):
    num_hidden: int
    num_heads: int
    invariant: BaseInvariant
    embedding_type: str
    embedding_freq_multiplier: tuple
    condition_value_transform: bool
    condition_invariant_embedding: bool
    project_heads: bool
    use_gaussian_window: bool = True

    def setup(self):
        # Get the invariant embedding.
        embedding_freq_multiplier_inv, embedding_freq_multiplier_value = self.embedding_freq_multiplier
        self.invariant_embedding_query = get_embedding(
            embedding_type=self.embedding_type,
            num_in=self.invariant.dim,
            num_hidden=self.num_hidden,
            num_emb_dim=self.num_hidden,
            freq_multiplier=embedding_freq_multiplier_inv
        )
        self.invariant_embedding_value = get_embedding(
            embedding_type=self.embedding_type,
            num_in=self.invariant.dim,
            num_hidden=self.num_hidden,
            num_emb_dim=self.num_hidden,
            freq_multiplier=embedding_freq_multiplier_value
        )

        # Setup linear layers
        self.inv_emb_to_q = nn.Dense(self.num_heads * self.num_hidden)
        self.a_to_k = nn.Dense(self.num_heads * self.num_hidden)
        self.a_to_v = nn.Dense(self.num_heads * self.num_hidden)

        # Set the scale factor for the attention weights.
        self.scale = 1.0 / (self.num_hidden ** 0.5)

        # If true create conditioning layers
        if self.condition_invariant_embedding:
            self.inv_emb_cond_to_inv_emb = PointwiseFFN(self.num_hidden, self.num_hidden, 2 * self.num_hidden)
        if self.condition_value_transform:
            self.inv_emb_to_v = PointwiseFFN(self.num_hidden, self.num_hidden, 2 * self.num_heads * self.num_hidden)
            self.inv_emb_cond_mixer = PointwiseFFN(self.num_hidden, self.num_hidden, self.num_hidden)

        # Define the projection heads, if num_heads > 0 than
        if self.project_heads:
            self.out_proj = nn.Dense(self.num_hidden)
        else:
            self.out_proj = nn.Dense(self.num_heads * self.num_hidden)

    def __call__(self, x, p, a, window_sigma=None, x_h=None):
        """ Apply equivariant cross attention.

        Args:
            x (jax.numpy.ndarray): The input coordinates. Shape (batch_size, num_coords, coord_dim).
            p (jax.numpy.ndarray): The latent coordinates. Shape (batch_size, num_latents, coord_dim).
            a (jax.numpy.ndarray): The latent features. Shape (batch_size, num_latents, latent_dim).
            window_sigma (jax.numpy.ndarray): The window size for the gaussian window. Shape (batch_size, num_latents, 1).
            x_h (jax.numpy.ndarray): The conditioning input. Shape (batch_size, num_coords, num_hidden). This is only used if condition_query_transform is True.
        """
        # Get invariants of input coordinates wrt latent coordinates. Depending on the invariant, the shape of the
        # invariants tensor will be different.
        inv = self.invariant(x, p)

        # Apply invariant embedding for the query transform.
        inv_emb_q = self.invariant_embedding_query(inv)

        # Calculate the query, key and value.
        q = self.inv_emb_to_q(inv_emb_q)
        k = self.a_to_k(a)
        v = self.a_to_v(a)

        # Optionally apply conditioning to the value transform based on the query. Broadcast v over the coordinates.
        if self.condition_value_transform:

            # Apply invariant embedding for the value transform conditioning.
            inv_emb_v = self.invariant_embedding_value(inv)

            # Optionally, if this is not the first cross-attention layer, apply
            if self.condition_invariant_embedding:
                assert x_h is not None, "cond_x must be provided if condition_query_transform is True."
                inv_emb_v_gamma_beta = self.inv_emb_cond_to_inv_emb(x_h)
                inv_emb_v_gamma, inv_emb_v_beta = jnp.split(inv_emb_v_gamma_beta, 2, axis=-1)

                # Apply conditioning to the query, broadcast over the latent points.
                inv_emb_v = inv_emb_v * (1 + inv_emb_v_gamma[:, :, None, :]) + inv_emb_v_beta[:, :, None, :]

            # Get conditioning variables for the value transform.
            v_gamma_beta = self.inv_emb_to_v(inv_emb_v)

            # Split gamma and beta conditioning variables
            v_gamma, v_beta = jnp.split(v_gamma_beta, 2, axis=-1)

            # Apply conditioning to the value transform, broadcast over the coordinates.
            v = v[:, None, :, :] * (1 + v_gamma) + v_beta

            # Reshape to separate the heads
            v = v.reshape(v.shape[:-1] + (self.num_heads, self.num_hidden))
            v = self.inv_emb_cond_mixer(v)
        else:
            v = v[:, None, :, :]

            # Reshape to separate the heads
            v = v.reshape(v.shape[:-1] + (self.num_heads, self.num_hidden))

        # Reshape the query, key and value to separate the heads.
        q = q.reshape(q.shape[:-1] + (self.num_heads, self.num_hidden))
        k = k.reshape(k.shape[:-1] + (self.num_heads, self.num_hidden))

        # For every point, calculate the attention weights for every latent point.
        att = (q * k[:, None, ...]).sum(axis=-1) * self.scale  # 'bczhd,bzhd->bczh'

        # Apply gaussian window if needed
        if self.use_gaussian_window:
            gaussian_window = self.invariant.calculate_gaussian_window(x, p, sigma=window_sigma)
            att = att + gaussian_window

        att = softmax(att, axis=-2)

        # Apply attention to the values.
        y = (att[..., None] * v).sum(axis=2)  # 'bczh,bczhd->bchd'

        # Reshape y to concatenate the heads.
        y = y.reshape(*y.shape[:2], self.num_heads * self.num_hidden)

        # output projection
        y = self.out_proj(y)
        return y


if __name__ == '__main__':
    from enf.steerable_attention.invariant.norm_rel_pos import NormRelativePositionND

    key = jax.random.PRNGKey(0)

    # Variables
    coord_dim = 2
    latent_dim = 5
    num_latents = 4
    num_signals = 3

    # Setup random inputs
    x = jax.random.normal(key, (num_signals, 100, coord_dim))
    p = jax.random.normal(key, (num_signals, num_latents, coord_dim))
    a = jax.random.normal(key, (num_signals, num_latents, latent_dim))
    window_sigma = jnp.ones((num_signals, num_latents, 1))

    x_h = jax.random.normal(key, (num_signals, 100, 256))

    # Setup equivariant cross attention layer
    invariant = NormRelativePositionND(coord_dim)
    m = EquivariantCrossAttention(
        num_hidden=256,
        num_heads=2,
        invariant=invariant,
        embedding_type='rff',
        embedding_freq_multiplier=(0.5, 10),
        condition_value_transform=True,
        condition_invariant_embedding=True,
        project_heads=True,
    )
    params = m.init(key, x, p, a, window_sigma, x_h)

    # Forward
    y = m.apply(params, x, p, a, window_sigma, x_h)
