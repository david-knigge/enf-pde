import jax.numpy as jnp

from enf.steerable_attention.invariant._base_invariant import BaseInvariant


class RelativePosition2DPeriodic(BaseInvariant):

    def __init__(self, num_dims: int):
        """ Calculate the relative position between two sets of coordinates in N dimensions, taking into account
        periodicity of the domain. Assumes that the domain is periodic in all dimensions over the range [-1, 1].

        Args:
            num_dims (int): The dimensionality of the coordinates, corresponds to the dimensionality of the translation
                group.
        """
        super().__init__()

        # Set the dimensionality of the invariant, since the domain is periodic, the dimensionality of the invariant
        # is twice the dimensionality of the coordinates as the coordinates are embedded into the complex plane.
        self.dim = 2 * num_dims

        # This invariant is calculated based on two sets of positional coordinates, it doesn't depend on
        # the orientation.
        self.num_x_pos_dims = num_dims
        self.num_x_ori_dims = 0
        self.num_z_pos_dims = num_dims
        self.num_z_ori_dims = 0

        # Set as periodic
        self.is_periodic = True

        # Set function to calculate the gaussian window
        self.calculate_gaussian_window = self._calculate_gaussian_window_periodic

    def __call__(self, x, p):
        """ Calculate the relative position between two sets of coordinates, taking into account periodicity of the
        domain. The shortest distance between two points in a periodic domain is calculated.

        Args:
            x (jnp.ndarray): The input coordinates. Shape (batch_size, num_coords, dim).
            p (jnp.ndarray): The latent coordinates. Shape (batch_size, num_latents, dim).
        Returns:
            invariants (torch.Tensor): The relative position between the input and latent coordinates.
                Shape (batch_size, num_coords, num_latents, dim).
        """

        # Calculate the relative position between the input and latent coordinates
        # [batch_size, num_coords, num_latents, 2]
        rel_pos_r2 = p[:, None, :, :] - x[:, :, None, :]

        # [batch_size, num_coords, num_latents, 4]
        invariants = jnp.concatenate(
            [
                jnp.cos(jnp.pi * rel_pos_r2),
                jnp.sin(jnp.pi * rel_pos_r2)
            ],
            axis=-1
        )

        return invariants


def apply_gaussian(x, p, sigma):
    # Calculate norm distance between x and p
    norm_rel_dists = -jnp.norm(
        jnp.cos(1/2 * jnp.pi * (p[:, None, :, :] - x[:, :, None, :])),
        keepdim=True,
        axis=-1,
        ord=2
    )

    # Calculate the gaussian window
    return - (1 / sigma ** 2) * norm_rel_dists


if __name__ == "__main__":
    import jax
    # Test the relative position invariant
    rel_pos = RelativePosition2DPeriodic(num_dims=2)

    meshgrid_over_domain = jnp.stack(jnp.meshgrid(
        jnp.linspace(-1, 1, 32), jnp.linspace(-1, 1, 32)), axis=-1).reshape(-1, 2)[None, ...]
    p = jnp.array([[[[0, 0]]]])
    pt = jnp.array([[[[0.5, 0.5]]]])

    # linear map from r4 to r1
    W = jax.random.normal(jax.random.PRNGKey(1), (4, 1))

    invariants = rel_pos(meshgrid_over_domain, p)
    invariants_t = rel_pos(meshgrid_over_domain, pt)
    inv_sum = invariants @ W
    inv_sum_t = invariants_t @ W

    import matplotlib.pyplot as plt
    plt.imshow(inv_sum.reshape(32, 32))
    plt.xticks([0, 15.5, 31], [-1, 0, 1])
    plt.yticks([0, 15.5, 31], [-1, 0, 1])
    plt.show()
    plt.imshow(inv_sum_t.reshape(32, 32))
    plt.xticks([0, 15.5, 31], [-1, 0, 1])
    plt.yticks([0, 15.5, 31], [-1, 0, 1])
    plt.show()

