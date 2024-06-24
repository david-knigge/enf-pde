from typing import Union
from functools import partial

import jax
import jax.numpy as jnp
from jax.nn import gelu, softmax
from flax import linen as nn


from enf.steerable_attention.invariant._base_invariant import BaseInvariant
from enf.steerable_attention.equivariant_cross_attention import EquivariantCrossAttention, PointwiseFFN


class EquivariantSelfAttentionBlock(nn.Module):
    """ Cross attention layer for the latent points, conditioned on the poses.

        Args:
            num_hidden (int): The number of hidden units.
            num_heads (int): The number of attention heads.
            attn_operator (EquivariantCrossAttention): The attention operator to use.
        """
    num_hidden: int
    num_heads: int
    attn_operator: EquivariantCrossAttention
    residual: bool
    project_heads: bool

    def setup(self):
        # Layer normalization
        self.layer_norm_attn = nn.LayerNorm()

        # Attention layer
        self.attn = self.attn_operator(num_hidden=self.num_hidden, num_heads=self.num_heads, project_heads=self.project_heads)

        # Pointwise feedforward network
        if self.project_heads:
            self.pointwise_ffn = PointwiseFFN(num_in=self.num_hidden, num_hidden=self.num_hidden, num_out=self.num_hidden)
        else:
            self.pointwise_ffn = PointwiseFFN(num_in=self.num_heads * self.num_hidden,
                                              num_hidden=self.num_heads * self.num_hidden,
                                              num_out=self.num_heads * self.num_hidden)
    
    def __call__(self, p, a, window_size):
        """ Perform attention over the latent points, conditioned on the poses.

        Args:
            x (torch.Tensor): The poses. Shape [(]batch_size, num_coords, 2].
            p (torch.Tensor): The poses of the latent points. Shape [num_latents, num_ori, 4].
            a (torch.Tensor): The features of the latent points. Shape [num_latents, num_ori, num_hidden].
            x_h (torch.Tensor): The conditional input. Shape [batch_size, num_coords, num_hidden].
            window_size (float): The window size for the gaussian window.
        """

        # Apply layer normalization to 'a'
        a_norm = self.layer_norm_attn(a)

        # Apply attention
        a_attn = self.attn(x=p, p=p, a=a_norm, x_h=a_norm, window_sigma=window_size)

        # Apply residual connection if specified
        if self.residual:
            a_res = a + a_attn
            a_out = self.pointwise_ffn(a_res)
        else:
            a_out = self.pointwise_ffn(a_attn)
        return a_out


class EquivariantTransformer(nn.Module):
    """ Equivariant self attention network for the latent points, conditioned on the poses.

    Args:
        num_hidden (int): The number of hidden units.
        num_heads (int): The number of attention heads.
        num_layers (int): The number of self-attention layers.
        num_out (int): The number of output coordinates.
        self_attn_invariant (BaseInvariant): The invariant to use for the attention operation.
        embedding_type (str): The type of embedding to use. 'rff' or 'siren'.
        embedding_freq_multiplier (Union[float, float]): The frequency multiplier for the embedding.
        condition_value_transform (bool): Whether to condition the value transform on the invariant.
    """

    num_hidden: int
    num_heads: int
    num_layers: int
    num_out: int
    self_attn_invariant: BaseInvariant
    embedding_type: str
    embedding_freq_multiplier: Union[float, float]
    condition_value_transform: bool
    global_pooling: bool = False

    def setup(self):
        
        # Create a constructor for the equivariant cross attention operation.
        equivariant_self_attn = partial(
            EquivariantCrossAttention,
            invariant=self.self_attn_invariant,
            embedding_type=self.embedding_type,
            embedding_freq_multiplier=self.embedding_freq_multiplier,
            condition_value_transform=self.condition_value_transform,
            condition_invariant_embedding=True
        )

        # Create network
        self_attention_blocks = []
        self.activation = gelu

        # Add code->latent stem
        self.latent_stem = nn.Dense(self.num_hidden)

        # Add latent self-attention blocks
        for i in range(self.num_layers):
            self_attention_blocks.append(
                EquivariantSelfAttentionBlock(
                    num_hidden=self.num_hidden,
                    num_heads=self.num_heads,
                    attn_operator=equivariant_self_attn,
                    residual=True,
                    project_heads=True
                )
            )
        self.self_attention_blocks = self_attention_blocks

        # Output layer
        self.out_proj = nn.Sequential([
            nn.Dense(self.num_hidden),
            gelu,
            nn.Dense(self.num_hidden),
            gelu,
            nn.Dense(self.num_out)
        ])

    def __call__(self, Z):
        """ Sample from the model.

        Args:
            Z: Set of latent variables. Tuple containing the poses, features and gaussian window size.
        """
        p, a, gaussian_window_size = Z

        # p contains angles, so we need to embed them into a circle.
        if self.self_attn_invariant.num_z_ori_dims > 0:
            p_pos, p_angles = p[:, :, :2], p[:, :, 2:]
            p = jnp.concatenate((p_pos, jnp.cos(p_angles), jnp.sin(p_angles)), axis=-1)

        # Map code to latent space.
        a = self.latent_stem(a)

        # Self+cross attention layers.
        for i in range(0, self.num_layers):
            # Apply self attention between latents.
            a = self.self_attention_blocks[i](p, a, window_size=None)
            a = self.activation(a)

        # If global pooling is enabled, apply global pooling.
        if self.global_pooling:
            a = jnp.max(a, axis=1, keepdims=False)

        # Output layer
        out = self.out_proj(a)

        return out


if __name__ == '__main__':
    #  Define invariant
    from enf.steerable_attention.invariant.rel_pos import RelativePositionND

    invariant = RelativePositionND(num_dims=2)
    num_hidden = 256
    num_heads = 4
    num_layers = 3
    num_out = 3

    # Define the model
    model = EquivariantTransformer(
        num_hidden=num_hidden,
        num_heads=num_heads,
        num_layers=num_layers,
        num_out=num_out,
        self_attn_invariant=invariant,
        embedding_type='rff',
        embedding_freq_multiplier=(0.5, 10),
        condition_value_transform=True,
        global_pooling=True
    )

    # Define the poses and features
    p = jnp.stack(jnp.meshgrid(jnp.linspace(-1, 1, 4), jnp.linspace(-1, 1, 4), indexing='ij'), axis=-1).reshape(-1, 2)
    p = jnp.broadcast_to(p, (1, *p.shape))

    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (1, 16, 128))
    gaussian_window_size = jnp.ones((1, 16, 1))

    # Initialize the model
    params = model.init(jax.random.PRNGKey(0), p, a, gaussian_window_size)

    # Apply the model
    out = model.apply(params, p, a, gaussian_window_size)
    print(out.shape)  # (1, 100, 3)
