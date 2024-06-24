from typing import Union
from functools import partial

import jax
import jax.numpy as jnp
from jax.nn import gelu, softmax
from flax import linen as nn

from enf.steerable_attention.invariant._base_invariant import BaseInvariant
from enf.steerable_attention.equivariant_cross_attention import EquivariantCrossAttention, PointwiseFFN


class EquivariantCrossAttentionBlock(nn.Module):
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
        self.attn = self.attn_operator(num_hidden=self.num_hidden, num_heads=self.num_heads,
                                       project_heads=self.project_heads)

        # Pointwise feedforward network
        if self.project_heads:
            self.pointwise_ffn = PointwiseFFN(num_in=self.num_hidden, num_hidden=self.num_hidden,
                                              num_out=self.num_hidden)
        else:
            self.pointwise_ffn = PointwiseFFN(num_in=self.num_heads * self.num_hidden,
                                              num_hidden=self.num_heads * self.num_hidden,
                                              num_out=self.num_heads * self.num_hidden)

    def __call__(self, x, p, a, x_h, window_size):
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
        a_attn = self.attn(x=x, p=p, a=a_norm, x_h=x_h, window_sigma=window_size)

        # Apply residual connection if specified
        if self.residual:
            a_res = a + a_attn
            a_out = self.pointwise_ffn(a_res)
        else:
            a_out = self.pointwise_ffn(a_attn)
        return a_out


class EquivariantCrossAttentionNeF(nn.Module):
    """ Equivariant cross attention network for the latent points, conditioned on the poses.

    Args:
        num_hidden (int): The number of hidden units.
        num_heads (int): The number of attention heads.
        num_layers (int): The number of self-attention layers.
        num_out (int): The number of output coordinates.
        latent_dim (int): The dimensionality of the latent code.
        invariant (BaseInvariant): The invariant to use for the attention operation.
        embedding_type (str): The type of embedding to use. 'rff' or 'siren'.
        embedding_freq_multiplier (Union[float, float]): The frequency multiplier for the embedding.
        condition_value_transform (bool): Whether to condition the value transform on the invariant.
    """

    num_hidden: int
    num_heads: int
    num_layers: int
    num_out: int
    latent_dim: int
    cross_attn_invariant: BaseInvariant
    self_attn_invariant: BaseInvariant
    embedding_type: str
    embedding_freq_multiplier: Union[float, float]
    condition_value_transform: bool
    cross_attention_blocks = []
    use_gaussian_window: bool = True

    def setup(self):

        # Create a constructor for the equivariant cross attention operation.
        equivariant_cross_attn = partial(
            EquivariantCrossAttention,
            invariant=self.cross_attn_invariant,
            embedding_type=self.embedding_type,
            embedding_freq_multiplier=self.embedding_freq_multiplier,
            condition_value_transform=self.condition_value_transform,
            condition_invariant_embedding=False,
            use_gaussian_window=self.use_gaussian_window
        )
        # equivariant_cond_cross_attn = partial(
        #     EquivariantCrossAttention,
        #     invariant=self.cross_attn_invariant,
        #     embedding_type=self.embedding_type,
        #     embedding_freq_multiplier=self.embedding_freq_multiplier,
        #     condition_value_transform=self.condition_value_transform,
        #     condition_invariant_embedding=True
        # )
        equivariant_self_attn = partial(
            EquivariantCrossAttention,
            invariant=self.self_attn_invariant,
            embedding_type=self.embedding_type,
            embedding_freq_multiplier=self.embedding_freq_multiplier,
            condition_value_transform=self.condition_value_transform,
            condition_invariant_embedding=False,
            use_gaussian_window=self.use_gaussian_window
        )

        # Create network
        cross_attention_blocks = []
        self_attention_blocks = []
        self.activation = gelu

        # Add code->latent stem
        self.latent_stem = nn.Dense(self.num_hidden)

        # Add latent self-attention blocks
        for i in range(self.num_layers):
            # # First layer attention is non-conditional
            # if i == 0:
            #     cross_attention_blocks.append(
            #         EquivariantCrossAttentionBlock(
            #             num_hidden=self.num_hidden,
            #             num_heads=self.num_heads,
            #             attn_operator=equivariant_cross_attn,
            #             residual=False,
            #             project_heads=True
            #         )
            #     )
            # else:
            #     cross_attention_blocks.append(
            #         EquivariantCrossAttentionBlock(
            #             num_hidden=self.num_hidden,
            #             num_heads=self.num_heads,
            #             attn_operator=equivariant_cond_cross_attn,
            #             residual=False,
            #             project_heads=True
            #         )
            #     )
            self_attention_blocks.append(
                EquivariantCrossAttentionBlock(
                    num_hidden=self.num_hidden,
                    num_heads=self.num_heads,
                    attn_operator=equivariant_self_attn,
                    residual=True,
                    project_heads=True
                )
            )

        # If there is only one layer, we need to add the final cross attention block that doesn't take conditioning
        # if self.num_layers == 0:
        cross_attention_blocks.append(
            EquivariantCrossAttentionBlock(
                num_hidden=self.num_hidden,
                num_heads=self.num_heads,
                attn_operator=equivariant_cross_attn,
                residual=False,
                project_heads=False
            )
        )
        # else:
        #     # Add final cross attention block and activation
        #     cross_attention_blocks.append(
        #         EquivariantCrossAttentionBlock(
        #             num_hidden=self.num_hidden,
        #             num_heads=self.num_heads,
        #             attn_operator=equivariant_cond_cross_attn,
        #             residual=False,
        #             project_heads=False
        #         )
        #     )

        self.cross_attention_blocks = cross_attention_blocks
        self.self_attention_blocks = self_attention_blocks

        # Output layer
        self.out_proj = nn.Sequential([
            nn.Dense(self.num_hidden),
            gelu,
            nn.Dense(self.num_hidden),
            gelu,
            nn.Dense(self.num_out)
        ])

    def __call__(self, x, p, a, gaussian_window_size):
        """ Sample from the model.

        Args:
            x (torch.Tensor): The pose of the input points. Shape (batch_size, num_coords, 2).
            p (torch.Tensor): The pose of the latent points. Shape (batch_size, num_latents, num_ori (1), 4).
            a (torch.Tensor): The latent features. Shape (batch_size, num_latents, num_hidden).
            gaussian_window_size (float or None): The window size for the gaussian window.
        """
        # p contains angles, so we need to embed them into a circle.
        if self.cross_attn_invariant.num_z_ori_dims > 0:
            p_pos, p_angles = p[:, :, :self.cross_attn_invariant.num_z_pos_dims], p[:, :,
                                                                                  self.cross_attn_invariant.num_z_pos_dims:]
            p = jnp.concatenate((p_pos, jnp.cos(p_angles), jnp.sin(p_angles)), axis=-1)

        # Map code to latent space.
        a = self.latent_stem(a)

        # Self attention layers.
        for i in range(0, self.num_layers):
            # Apply self attention between latents.
            a = a + self.self_attention_blocks[i](p, p, a, x_h=None, window_size=gaussian_window_size)
            a = self.activation(a)

        # Final cross attention block
        out = self.cross_attention_blocks[-1](x, p, a, x_h=None, window_size=gaussian_window_size)
        out = self.activation(out)

        # Output layer
        out = self.out_proj(out)

        return out


if __name__ == "__main__":
    # Define the model
    model = EquivariantCrossAttentionNeF(
        num_hidden=256,
        num_heads=8,
        num_layers=3,
        num_out=2,
        latent_dim=4,
        cross_attn_invariant=BaseInvariant(num_z_pos_dims=2, num_z_ori_dims=2),
        self_attn_invariant=BaseInvariant(num_z_pos_dims=2, num_z_ori_dims=2),
        embedding_type='rff',
        embedding_freq_multiplier=2.0,
        condition_value_transform=True
    )

    # Define the input
    x = jnp.ones((2, 10, 2))
    p = jnp.ones((2, 5, 1, 4))
    a = jnp.ones((2, 5, 256))
    gaussian_window_size = 0.1

    # Apply the model
    out = model(x, p, a, gaussian_window_size)
    print(out.shape)  # (2, 10, 2)
