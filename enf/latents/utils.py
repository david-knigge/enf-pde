import jax.numpy as jnp


def init_positions_ball(rng, shape):
    """
    Initialize the latent poses on a grid of polar coordinates using JAX.

    Args:


    Returns:
        z_positions (jax.numpy.ndarray): The latent poses for each signal. Shape [num_signals, num_latents, ...].
    """
    num_signals, num_latents, _ = shape

    # Create a grid of euler angles (alpha, beta gamma) based on fibonacci lattice
    indices = jnp.arange(1, num_latents + 1)
    alpha = jnp.arccos(1 - 2 * indices / (num_latents + 1))
    beta = jnp.pi * (1 + 5 ** 0.5) * indices

    # Set roll as based on the number of latents
    gamma = jnp.arange(0, 2 * jnp.pi, 2 * jnp.pi / num_latents)

    # Stack and reshape to create the positions matrix
    positions = jnp.stack([alpha, beta, gamma], axis=-1).reshape(-1, 3)

    # Add radius
    positions = jnp.concatenate([positions, jnp.ones((positions.shape[0], 1)) * 0.75], axis=-1)

    # Repeat for the number of signals
    positions = jnp.repeat(positions[None, :, :], num_signals, axis=0)

    return positions


def init_positions_polar(rng, shape):
    """
    Initialize the latent poses on a grid of polar coordinates using JAX.

    Args:


    Returns:
        z_positions (jax.numpy.ndarray): The latent poses for each signal. Shape [num_signals, num_latents, ...].
    """
    num_signals, num_latents, num_dims = shape

    # Assert num_dims = 2, since we are working with polar coordinates
    num_latents = num_latents // 2

    # Ensure num_latents is a power of num_dims
    assert abs(round(num_latents ** (1. / num_dims), 5) % 1) < 1e-5, 'num_latents must be a power of the number of position dimensions'

    # Calculate the number of latents per dimension
    num_latents_per_dim = int(round(num_latents ** (1. / num_dims)))

    # Create an n-dimensional mesh grid [-1 to 1] for each dimension
    grid_phi = jnp.linspace(0 + jnp.pi / (2*num_latents_per_dim), 2 * jnp.pi - jnp.pi / (2 * num_latents_per_dim),
                            2 * num_latents_per_dim)
    grid_theta = jnp.linspace(0 + (jnp.pi / 2) / num_latents_per_dim, jnp.pi - (jnp.pi / 2) / num_latents_per_dim, num_latents_per_dim)

    grids = jnp.meshgrid(grid_phi, grid_theta, indexing='ij')

    # Stack and reshape to create the positions matrix
    positions = jnp.stack(grids, axis=-1).reshape(-1, num_dims)

    # Repeat for the number of signals
    positions = jnp.repeat(positions[None, :, :], num_signals, axis=0)

    return positions


def init_positions_grid(rng, shape):
    """
    Initialize the latent poses on a grid using JAX.

    Args:
        num_latents (int): The number of latent points.
        num_signals (int): The number of signals.
        num_dims (int): The number of dimensions for each point.

    Returns:
        z_positions (jax.numpy.ndarray): The latent poses for each signal. Shape [num_signals, num_latents, ...].
    """
    num_signals, num_latents, num_dims = shape

    # Ensure num_latents is a power of num_dims
    assert abs(round(num_latents ** (1. / num_dims), 5) % 1) < 1e-5, 'num_latents must be a power of the number of position dimensions'

    # Calculate the number of latents per dimension
    num_latents_per_dim = int(round(num_latents ** (1. / num_dims)))

    # Create an n-dimensional mesh grid [-1 to 1] for each dimension
    grid_axes = jnp.linspace(-1 + 1 / num_latents_per_dim, 1 - 1 / num_latents_per_dim, num_latents_per_dim)
    grids = jnp.meshgrid(*[grid_axes for _ in range(num_dims)], indexing='ij')

    # Stack and reshape to create the positions matrix
    positions = jnp.stack(grids, axis=-1).reshape(-1, num_dims)

    # Repeat for the number of signals
    positions = jnp.repeat(positions[None, :, :], num_signals, axis=0)

    return positions


def init_ori_rotation_invariant_s2(rng, shape):
    """ Only initializes a single orientation, based on . """
    positions = init_positions_grid(None, shape)
    return jnp.arctan2(positions[:, :, 0], positions[:, :, 1])[:, :, None]


def init_appearances_ones_jax(num_latents: int, num_signals: int, latent_dim: int):
    """
    Initialize the latent features as ones using JAX.

    Args:
        num_latents (int): The number of latent points.
        num_signals (int): The number of signals.
        latent_dim (int): The dimensionality of the latent code.

    Returns:
        z_features (jax.numpy.ndarray): The latent features for each signal. Shape [num_signals, num_latents, latent_dim].
    """
    z_features = jnp.ones((num_signals, num_latents, latent_dim))
    return z_features


def init_orientations_fixed(num_latents:int, num_signals: int, num_dims: int):
    """ Initialize the latent orientations as fixed.

    Args:
        num_latents (int): The number of latent points.
        num_signals (int): The number of signals.

    Returns:
        z_orientations (torch.Tensor): The latent orientations for each signal. Shape [num_signals, num_latents, ...].
    """
    orientations = jnp.zeros((num_signals, num_latents, num_dims))
    return orientations