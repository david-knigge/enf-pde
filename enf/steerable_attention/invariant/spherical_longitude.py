import jax.numpy as jnp

from enf.steerable_attention.invariant._base_invariant import BaseInvariant


class RelativeLatitudePeriodic(BaseInvariant):

    def __init__(self):
        """ 

        Args:
            num_dims (int): The dimensionality of the coordinates, corresponds to the dimensionality of the translation
                group.
        """
        super().__init__()

        # Set the dimensionality of the invariant, since the domain is periodic, the dimensionality of the invariant
        # is twice the dimensionality of the coordinates as the coordinates are embedded into the complex plane.
        self.dim = 4

        # This invariant is calculated based on two sets of positional coordinates, it doesn't depend on
        # the orientation.
        self.num_x_pos_dims = 2
        self.num_x_ori_dims = 0
        self.num_z_pos_dims = 2
        self.num_z_ori_dims = 0

        # Set as periodic
        self.is_periodic = True

        # Set function to calculate the gaussian window
        self.calculate_gaussian_window = self._calculate_gaussian_window_periodic_sphere

    def _calculate_gaussian_window_periodic_sphere(self, x, p, sigma):
        """ Calculate gaussian window for sphere. """
        
        # Get lat and lon
        # Get lat and lon
        phi_x = x[:, :, 0]
        theta_x = x[:, :, 1]

        phi_p = p[:, :, 0]
        theta_p = p[:, :, 1]

        # Convert polar to cartesian
        x = jnp.stack([jnp.sin(theta_x) * jnp.cos(phi_x), jnp.sin(theta_x) * jnp.sin(phi_x), jnp.cos(theta_x)], axis=-1)
        p = jnp.stack([jnp.sin(theta_p) * jnp.cos(phi_p), jnp.sin(theta_p) * jnp.sin(phi_p), jnp.cos(theta_p)], axis=-1)

        # Calculate the cosine similarity
        ang = jnp.einsum('bnd,bmd->bnm', x, p)[:, :, :, None] / (jnp.linalg.norm(x, axis=-1)[:, :, None, None] * jnp.linalg.norm(p, axis=-1)[:, None, :, None])

        # Return arccos
        dist = jnp.arccos(jnp.clip(ang, -1 + 1e-6, 1 - 1e-6))

        return jnp.exp(-dist**2 / (2 * sigma[:, None, :, :] ** 2))

    def __call__(self, x, p):
        """ Calculate the relative position between two sets of coordinates, taking into account periodicity of the
        domain. The shortest distance between two points in a periodic domain is calculated.

        Args:
            x (jnp.ndarray): The input coordinates. Shape (batch_size, num_coords, dim). In polar coordinates.
            p (jnp.ndarray): The latent coordinates. Shape (batch_size, num_latents, dim). In polar coordinates.
        Returns:
            invariants (jnp.ndarray): The relative position between the input and latent coordinates.
                Shape (batch_size, num_coords, num_latents, dim).
        """
        # Get lat and lon
        phi_x = jnp.broadcast_to(x[:, :, None, 0], (x.shape[0], x.shape[1], p.shape[1]))[..., None]
        theta_x = jnp.broadcast_to(x[:, :, None, 1], (x.shape[0], x.shape[1], p.shape[1]))[..., None]

        phi_p = jnp.broadcast_to(p[:, None, :, 0], (p.shape[0], x.shape[1], p.shape[1]))[..., None]
        theta_p = jnp.broadcast_to(p[:, None, :, 1], (p.shape[0], x.shape[1], p.shape[1]))[..., None]

        invariants = jnp.concatenate(
            [   
                theta_x,
                theta_p,
                jnp.cos(phi_x - phi_p),
                jnp.sin(phi_x - phi_p)
            ],
            axis=-1
        )

        return invariants


if __name__ == "__main__":
    import jax
    # Example usage
    num_dims = 2
    relative_position = RelativeLatitudePeriodic()

    # Meshgrid of polar coordinates
    x = jnp.stack(jnp.meshgrid(jnp.linspace(0, 2 * jnp.pi, 256), jnp.linspace(0, jnp.pi, 128), indexing='ij'), axis=-1)

    W = jax.random.normal(jax.random.PRNGKey(0), (4, 1))

    # Reshape to (num_coords, num_dims)
    x = x.reshape(-1, 2)[None]
    p = jnp.array([[0 * jnp.pi, 0 * jnp.pi]])[None, :, :]

    for i in jnp.arange(0, jnp.pi, jnp.pi / 4):
        p = jnp.array([[i, 1.6]])[None, :, :]

        # Cosine sim
        invariants = relative_position(x, p)
        invariants = invariants @ W

        import matplotlib.pyplot as plt
        plt.imshow(invariants.reshape(256, 128))
        plt.show()

    for i in jnp.arange(0, jnp.pi, jnp.pi / 4):
        p = jnp.array([[1.6, i]])[None, :, :]

        # Cosine sim
        invariants = relative_position(x, p)
        invariants = invariants @ W

        import matplotlib.pyplot as plt

        plt.imshow(invariants.reshape(256, 128))
        plt.show()
        # Gaussian window
        # sigma = jnp.array([[[1.5]]])
        # gaussian_window = relative_position.calculate_gaussian_window(x, p, sigma)
        # plt.imshow(gaussian_window.reshape(256, 128))
        # plt.show()
