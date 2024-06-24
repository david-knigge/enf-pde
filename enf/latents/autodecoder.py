import jax.numpy as jnp
from flax import linen as nn
from functools import partial

from enf.latents.utils import init_positions_grid, init_positions_polar, init_ori_rotation_invariant_s2, init_positions_ball


class PositionOrientationFeatureAutodecoder(nn.Module):
    num_signals: int
    num_latents: int
    latent_dim: int
    num_pos_dims: int
    num_ori_dims: int
    gaussian_window_size: float = None  # Use None as the default for optional arguments
    frequency_parameter: float = None
    coordinate_system: str = 'cartesian'

    def setup(self):
        # Initialize the latent positions, orientations, appearances, gaussian window, and frequency parameter here
        if self.coordinate_system == 'cartesian':
            self.p_pos = self.param('p_pos', init_positions_grid, (self.num_signals, self.num_latents, self.num_pos_dims))
        elif self.coordinate_system == 'polar':
            self.p_pos = self.param('p_pos', init_positions_polar, (self.num_signals, self.num_latents, self.num_pos_dims))
        elif self.coordinate_system == 'ball':
            self.p_pos = self.param('p_pos', init_positions_ball, (self.num_signals, self.num_latents, self.num_pos_dims))

        if self.num_ori_dims > 0:
            assert self.num_pos_dims == 2, "Only implemented for 2D"
            self.p_ori = self.param('p_ori', init_ori_rotation_invariant_s2, (self.num_signals, self.num_latents, self.num_pos_dims))
        else:
            self.p_ori = None

        self.a = self.param('a', nn.initializers.ones, (self.num_signals, self.num_latents, self.latent_dim))

        # Calculate gaussian window size s.t. each gaussian overlaps.
        # This is the same as setting the standard deviation to the distance between the latent points.
        # Create a grid of latent positions
        if self.coordinate_system == 'cartesian':
            num_latents_per_dim = int(round(self.num_latents ** (1. / self.num_pos_dims), 5))

            # Since our domain ranges from -1 to 1, the distance between each latent point is 2 / num_latents_per_dim
            # We want each gaussian to be centered at a latent point, and be removed 2 std from other latent points.
            gaussian_window_size = self.num_pos_dims / num_latents_per_dim

        elif self.coordinate_system == 'polar': # In case of polar coords, we use twice res along longitudinal axis
            num_latents_per_dim = int(round((self.num_latents//2) ** (1. / self.num_pos_dims), 5))

            # In this setting our domain ranges from 0 to 2pi, the distance between each latent point is
            # 2pi / num_latents_per_dim
            # We want each gaussian to be centered at a latent point, and be removed 2 std from other latent points.
            gaussian_window_size = self.num_pos_dims * jnp.pi / num_latents_per_dim

        elif self.coordinate_system == 'ball':
            gaussian_window_size = 1.0

        self.gaussian_window = self.param('gaussian_window', nn.initializers.constant(gaussian_window_size), (self.num_signals, self.num_latents, 1))

    def __call__(self, idx: int):
        # Implement the forward pass using JAX operations
        p_pos = self.p_pos[idx]

        if self.num_ori_dims > 0:
            p_ori = self.p_ori[idx]
            p = jnp.concatenate((p_pos, p_ori), axis=-1)
        else:
            p = p_pos

        a = self.a[idx]

        # Optionally, get the gaussian window for the latent points
        gaussian_window = self.gaussian_window[idx]

        return p, a, gaussian_window
