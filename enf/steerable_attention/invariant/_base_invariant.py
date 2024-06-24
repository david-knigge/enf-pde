import numpy as np 
import jax.numpy as jnp


class BaseInvariant:

    def __init__(self):
        super().__init__()

        # Every invariant has a dimensionality.
        self.dim = None

        # Every invariant can have a different number of dimensions for the input coordinate and latent poses.
        self.num_x_pos_dims = None
        self.num_x_ori_dims = None
        self.num_z_pos_dims = None
        self.num_z_ori_dims = None

        # Store periodicity flag, default to False
        self.is_periodic = False

        # Set function to calculate the gaussian window
        self.calculate_gaussian_window = self._calculate_gaussian_window_nonperiodic

    def _calculate_gaussian_window_nonperiodic(self, x, p, sigma):
        p_pos = p[:, :, :self.num_z_pos_dims]
        x_pos = x[:, :, :self.num_x_pos_dims]

        # Calculate squared norm distance between x and p
        norm_rel_dists = jnp.sum((p_pos[:, None, :, :] - x_pos[:, :, None, :]) ** 2, axis=-1, keepdims=True)

        # Calculate the gaussian window
        return - (1 / sigma[:, None, :] ** 2) * norm_rel_dists

    def _calculate_gaussian_window_periodic(self, x, p, sigma):
        p_pos = p[:, :, :self.num_z_pos_dims]
        x_pos = x[:, :, :self.num_x_pos_dims]

        # Calculate norm distance, considering periodicity
        norm_rel_dists = - jnp.sum(jnp.cos(np.pi * (p_pos[:, None, :, :] - x_pos[:, :, None, :])) ** 2, axis=-1, keepdims=True)

        # Calculate the gaussian window
        return - (1 / sigma[:, None, :] ** 2) * norm_rel_dists

    def __call__(self, x, p):
        raise NotImplementedError("Subclasses must implement this method")
