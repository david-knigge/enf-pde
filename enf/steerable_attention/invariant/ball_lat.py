import jax.numpy as jnp
import jax.random

from enf.steerable_attention.invariant._base_invariant import BaseInvariant


class BallLatInvariant(BaseInvariant):

    def __init__(self):
        """ Calculate the relative position between two sets of coordinates in N dimensions, taking into account
        periodicity of the domain. Assumes that the domain is periodic in all dimensions over the range [-1, 1].

        Args:
            num_dims (int): The dimensionality of the coordinates, corresponds to the dimensionality of the translation
                group.
        """
        super().__init__()

        # Set the dimensionality of the invariant, since the domain is periodic, the dimensionality of the invariant
        # is twice the dimensionality of the coordinates as the coordinates are embedded into the complex plane.
        self.dim = 6

        # This invariant is calculated based on two sets of positional coordinates, it doesn't depend on
        # the orientation.
        self.num_x_pos_dims = 3
        self.num_x_ori_dims = 0
        self.num_z_pos_dims = 4
        self.num_z_ori_dims = 0

        # Set as periodic
        self.is_periodic = False

        # Set function to calculate the gaussian window
        self.calculate_gaussian_window = self._calculate_gaussian_window_ball

    def _calculate_gaussian_window_ball(self, x, p, sigma):
        """ Calculate gaussian window for sphere. """
        # Get lat and lon, interpret euler angles alpha, beta as phi, theta
        phi_x = x[:, :, 0]
        theta_x = x[:, :, 1]

        phi_p = p[:, :, 0]
        theta_p = p[:, :, 1]

        # Convert polar to cartesian
        x = jnp.stack([jnp.sin(theta_x) * jnp.cos(phi_x), jnp.sin(theta_x) * jnp.sin(phi_x), jnp.cos(theta_x)], axis=-1)
        p = jnp.stack([jnp.sin(theta_p) * jnp.cos(phi_p), jnp.sin(theta_p) * jnp.sin(phi_p), jnp.cos(theta_p)], axis=-1)

        # Calculate the cosine similarity
        ang = jnp.einsum('bnd,bmd->bnm', x, p)[:, :, :, None] / (jnp.linalg.norm(x, axis=-1)[:, :, None, None] * jnp.linalg.norm(p, axis=-1)[:, None, :, None])
        dist = jnp.arccos(jnp.clip(ang, -1 + 1e-6, 1 - 1e-6))
        return jnp.exp(-dist ** 2 / (2 * sigma[:, None, :, :] ** 2))

    def __call__(self, x, p):
        """ Calculate the relative position between two sets of coordinates, taking into account periodicity of the
        domain. The shortest distance between two points in a periodic domain is calculated.

        Args:
            x (jnp.ndarray): The input coordinates. Shape (batch_size, num_coords, dim). In polar coordinates.
            p (jnp.ndarray): The latent coordinates. Shape (batch_size, num_latents, dim). In polar coordinates.
        Returns:
            invariants (torch.Tensor): The relative position between the input and latent coordinates.
                Shape (batch_size, num_coords, num_latents, dim).
        """
        # Get lat and lon
        phi_x = jnp.broadcast_to(x[:, :, None, 0], (x.shape[0], x.shape[1], p.shape[1]))[..., None]
        theta_x = jnp.broadcast_to(x[:, :, None, 1], (x.shape[0], x.shape[1], p.shape[1]))[..., None]

        phi_p = jnp.broadcast_to(p[:, None, :, 0], (p.shape[0], x.shape[1], p.shape[1]))[..., None]
        theta_p = jnp.broadcast_to(p[:, None, :, 1], (p.shape[0], x.shape[1], p.shape[1]))[..., None]

        r_x = x[:, :, 2]

        # Get euler angles
        r_p = p[:, :, 3]

        invariants = jnp.concatenate(
            [
                theta_x,
                theta_p,
                jnp.cos(phi_x - phi_p),
                jnp.sin(phi_x - phi_p),
                r_x[:, :, None, None],
                r_p[:, None, :, None],
            ],
            axis=-1
        )
        return invariants


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # Example usage
    num_dims = 3
    inv = BallInvariant()

    # Meshgrid of polar coordinates
    phi_grid = jnp.linspace(0, 2 * jnp.pi, 64)
    theta_grid = jnp.linspace(0, jnp.pi, 32)
    r_grid = jnp.linspace(0, 1, 48)
    phi, theta, r = jnp.meshgrid(phi_grid, theta_grid, r_grid, indexing='ij')
    x = jnp.stack([phi, theta, r], axis=-1)

    # Random map
    W = jax.random.normal(jax.random.PRNGKey(0), (3, 1))

    # Reshape to (num_coords, num_dims)
    x = x.reshape(-1, 3)[None]

    for i in jnp.arange(0, 2*jnp.pi, jnp.pi//2):
        p = jnp.array([[0.74 * jnp.pi, i, 0.75]])[None, :, :]

        # calc invariants
        invariants = inv(x, p)

        # Calc gaussian window
        sigma = jnp.array([[[0.5]]])
        gaussian_window = inv.calculate_gaussian_window(x, p, sigma).reshape(*phi.shape, 1)

        # reshape to ball grid
        inv_reshaped = invariants.reshape(*phi.shape, 3)

        # Select a slice
        phi_slice = phi[:, phi.shape[1]//2, :]
        theta_slice = theta[:, phi.shape[1]//2, :]
        r_slice = r[:, phi.shape[1]//2, :]
        inv_slice = inv_reshaped[:, phi.shape[1]//2, :]
        window_slice = gaussian_window[:, phi.shape[1]//2, :]

        # Select sphere
        phi_sphere = phi[:, phi.shape[1]//2:, -1]
        theta_sphere = theta[:, phi.shape[1]//2:, -1]
        r_sphere = r[:, phi.shape[1]//2:, -1]
        inv_sphere = inv_reshaped[:, phi.shape[1]//2:, -1]
        window_sphere = gaussian_window[:, phi.shape[1]//2:, -1]

        def spherical_to_cartesian(r, phi, theta):
            # Map to cartesian coordinates
            x = r * jnp.sin(theta) * jnp.cos(phi)
            y = r * jnp.sin(theta) * jnp.sin(phi)
            z = r * jnp.cos(theta)
            return x, y, z

        x_, y, z = spherical_to_cartesian(r_slice, phi_slice, theta_slice)

        x_sphere, y_sphere, z_sphere = spherical_to_cartesian(r_sphere, phi_sphere, theta_sphere)

        inv_sphere = inv_sphere @ W
        inv_ball = inv_slice @ W

        # Plot surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x_sphere, y_sphere, z_sphere, facecolors=plt.cm.magma(inv_sphere), rstride=1, cstride=1,
                        linewidth=1,
                        antialiased=False, shade=False)
        ax.plot_surface(x_, y, z, facecolors=plt.cm.magma(inv_ball), rstride=1, cstride=1, linewidth=0, antialiased=False,
                        shade=False)
        # Set limits
        ax.axes.set_xlim3d(left=-1.2, right=1.2)
        ax.axes.set_ylim3d(bottom=-1.2, top=1.2)
        ax.axes.set_zlim3d(bottom=-1.2, top=1.2)

        plt.savefig(f'ball-{i}.png', dpi=300)

        # Plot surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x_sphere, y_sphere, z_sphere, facecolors=plt.cm.magma(window_sphere), rstride=1, cstride=1,
                        linewidth=1,
                        antialiased=False, shade=False)
        ax.plot_surface(x_, y, z, facecolors=plt.cm.magma(window_slice), rstride=1, cstride=1, linewidth=0, antialiased=False,
                        shade=False)
        # Set limits
        ax.axes.set_xlim3d(left=-1.2, right=1.2)
        ax.axes.set_ylim3d(bottom=-1.2, top=1.2)
        ax.axes.set_zlim3d(bottom=-1.2, top=1.2)

        plt.show()
        plt.savefig(f'ball-{i}-win.png', dpi=300)

    # for i in jnp.arange(0, 2 * jnp.pi, jnp.pi / 4):
    #     p = jnp.array([[jnp.pi, i]])[None, :, :]
    #
    #     # Cosine sim
    #     invariants = relative_position(x, p)
    #
    #     import matplotlib.pyplot as plt
    #
    #     plt.imshow(invariants.reshape(256, 128))
    #     plt.show()
    #
    #     # Gaussian window
    #     sigma = jnp.array([[1.5]])
    #     gaussian_window = relative_position.calculate_gaussian_window(x, p, sigma)
    #     plt.imshow(gaussian_window.reshape(256, 128))
    #     plt.show()