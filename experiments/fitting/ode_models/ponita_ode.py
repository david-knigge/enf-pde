from typing import Union

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

from enf.steerable_attention.embedding.rff import RFFEmbedding


def chang_xavier_uniform(key, shape, dtype=jnp.float32):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    std = np.sqrt(2.0 / ((fan_in + fan_out)) * fan_in)  # hidden dim of network is same as hidden dim of kernel func
    return jax.random.uniform(key, shape, dtype, -std, std)


class PolynomialFeatures(nn.Module):
    degree: int

    def setup(self):
        pass  # No setup needed as there are no trainable parameters.

    def __call__(self, x):
        polynomial_list = [x]
        for it in range(1, self.degree+1):
            polynomial_list.append(jnp.einsum('...i,...j->...ij', polynomial_list[-1], x).reshape(*x.shape[:-1], -1))
        return jnp.concatenate(polynomial_list, axis=-1)


class GridGenerator(nn.Module):
    n: int
    dimension: int = 2
    steps: int = 200
    step_size: float = 0.01
    alpha: float = 0.001

    def setup(self):
        self.params = self.param('params', nn.initializers.zeros, (self.n, self.dimension + 1))

    def __call__(self):
        if self.dimension == 1:
            return self.uniform_grid_s1()
        elif self.dimension == 2:
            return self.repulse(self.random_s2())
        else:
            raise ValueError("Dimension must be either 1 (circle) or 2 (sphere).")

    def uniform_grid_s1(self):
        theta = jnp.linspace(0, 2 * jnp.pi, self.n+1)[:-1]
        x = jnp.cos(theta)
        y = jnp.sin(theta)
        return jnp.stack([x, y], axis=-1)

    def random_s2(self):
        points = random.normal(random.PRNGKey(0), (self.n, 3))
        points /= jnp.linalg.norm(points, axis=-1, keepdims=True)
        return points

    def repulse(self, points):
        points = jax.lax.stop_gradient(points)
        optimizer = sgd(self.step_size)
        opt_state = optimizer.init(points)

        def energy_loss(points):
            dists = jnp.sum(jnp.square(points[:, None] - points[None, :]), axis=-1)
            dists = jnp.linalg.norm(dists, axis=-1)
            dists = jax.lax.clamp(min=1e-6, x=dists, max=1e4)
            energy = jnp.sum(jnp.power(dists, -2))
            return energy
        
        for step in range(1, self.steps + 1):
            grads = jax.grad(energy_loss)(points)
            updates, opt_state = optimizer.update(grads, opt_state)
            points = optax.apply_updates(points, updates)
            # points /= jnp.linalg.norm(points, axis=-1, keepdims=True)
        return points


class SepConvNextBlock(nn.Module):
    num_hidden: int
    basis_dim: int
    widening_factor: int

    def setup(self):
        self.conv = SepConv(self.num_hidden, self.basis_dim)
        self.act_fn = nn.gelu
        self.linear_1 = nn.Dense(self.widening_factor * self.num_hidden)
        self.linear_2 = nn.Dense(self.num_hidden)
        self.norm = nn.LayerNorm()

    def __call__(self, x, kernel_basis, fiber_kernel_basis):
        x = self.conv(x, kernel_basis, fiber_kernel_basis)
        x = self.norm(x)
        x = self.linear_1(x)
        x = self.act_fn(x)
        x = self.linear_2(x)
        # x = x + input
        return x


class SepConv(nn.Module):
    num_hidden: int
    basis_dim: int
    bias: bool = True

    def setup(self):
        # Set up kernel coefficient layers, one for the spatial kernel and one for the group kernel.
        # This maps from the invariants to a basis for the kernel 2/3->basis_dim.
        self.spatial_kernel = nn.Dense(self.num_hidden, use_bias=False, kernel_init=chang_xavier_uniform)
        self.rotation_kernel = nn.Dense(self.num_hidden, use_bias=False, kernel_init=chang_xavier_uniform)

        # Construct bias
        if self.bias:
            self.bias_param = self.param('bias', nn.initializers.zeros, (self.num_hidden,))

    def __call__(self, x, kernel_basis, fiber_kernel_basis):
        """ Perform separable convolution on fully connected pointcloud.

        Args:
            x: Array of shape (batch, num_points, num_ori, num_features)
            kernel_basis: Array of shape (batch, num_points, num_points, num_ori, basis_dim)
            fiber_kernel_basis: Array of shape (batch, num_points, num_points, basis_dim)
        """
        # Compute the spatial kernel
        spatial_kernel = self.spatial_kernel(kernel_basis)

        # Compute the group kernel
        rot_kernel = self.rotation_kernel(fiber_kernel_basis)

        # Perform the convolution
        x = jnp.einsum('bnoc,bmnoc->bmoc', x, spatial_kernel)
        x = jnp.einsum('bmoc,poc->bmpc', x, rot_kernel)

        # Add bias
        if self.bias:
            x = x + self.bias_param
        return x


class PonitaFixedSize(nn.Module):
    num_hidden: int
    num_layers: int
    scalar_num_out: int
    vec_num_out: int

    invariant: nn.Module
    spatial_dim: int
    num_ori: int

    basis_dim: int
    degree: int
    widening_factor: int
    global_pool: bool
    last_feature_conditioning: bool = False
    kernel_size: Union[float, str] = "global"

    def setup(self):
        # Check input arguments
        assert self.spatial_dim in [2, 3], "spatial_dim must be 2 or 3."
        rot_group_dim = 1 if self.spatial_dim == 2 else 3
        assert self.kernel_size == "global" or self.kernel_size > 0, "kernel_size must be 'global' or a positive number."

        # Create a grid generator for the dimensionality of the orientation
        self.grid_generator = GridGenerator(n=self.num_ori, dimension=rot_group_dim, steps=1000)
        self.ori_grid = self.grid_generator()

        # Set up kernel basis functions, one for the spatial kernel and one for the group kernel.
        # This maps from the invariants to a basis for the kernel 2/3->basis_dim.
        self.spatial_kernel_basis = nn.Sequential([
            PolynomialFeatures(degree=self.degree), nn.Dense(self.num_hidden), nn.gelu, nn.Dense(self.basis_dim), nn.gelu])
        self.rotation_kernel_basis = nn.Sequential([
            PolynomialFeatures(degree=self.degree), nn.Dense(self.num_hidden), nn.gelu, nn.Dense(self.basis_dim), nn.gelu])

        # Initial node embedding
        self.x_embedder = nn.Dense(self.num_hidden, use_bias=False)

        # Make feedforward network
        interaction_layers = []
        for i in range(self.num_layers):
            interaction_layers.append(SepConvNextBlock(self.num_hidden, self.basis_dim, self.widening_factor))
        self.interaction_layers = interaction_layers

        # Readout layers
        self.readout = nn.Sequential([
            nn.Dense(
                self.scalar_num_out + self.vec_num_out,
                use_bias=False,
                kernel_init=nn.initializers.variance_scaling(1e-6, 'fan_in', 'truncated_normal')
            )
        ])

    def __call__(self, latent):
        """ Forward pass through the network.

        Args:
            pos: Array of shape (batch, num_points, spatial_dim)
            x: Array of shape (batch, num_points, num_in)
        """
        pos, x, _ = latent
        pos = pos[:, :, :self.spatial_dim]

        # Calculate invariants

        # Calculate invariants
        rel_pos = pos[:, None, :, None, :] - pos[:, :, None, None, :]  # (batch, num_points, num_points, 1, 3)
        invariant1 = (rel_pos[..., 0] * self.ori_grid[None, None, None, :, 0] + rel_pos[..., 1] * self.ori_grid[None, None, None, :, 1])[..., None]
        invariant2 = (-rel_pos[..., 0] * self.ori_grid[None, None, None, :, 1] + rel_pos[..., 1] * self.ori_grid[None, None, None, :, 0])[..., None]  # (batch, num_points, num_points, 1, 3)
        spatial_invariants = jnp.concatenate([invariant1, invariant2], axis=-1)
        orientation_invariants = (self.ori_grid[:, None, :] * self.ori_grid[None, :, :]).sum(axis=-1, keepdims=True)

        # Sample the kernel basis
        kernel_basis = self.spatial_kernel_basis(spatial_invariants)
        fiber_kernel_basis = self.rotation_kernel_basis(orientation_invariants)

        # Apply exponential window to spatial kernel based on rel_pos
        if self.kernel_size != "global":
            kernel_basis = kernel_basis * jnp.exp(-jnp.linalg.norm(rel_pos, axis=-1, keepdims=True) / self.kernel_size)

        # Initial feature embedding
        x = self.x_embedder(x)

        # Repeat features over the orientation dimension
        x = x[:, :, None, :].repeat(self.ori_grid.shape[-2], axis=-2) / self.ori_grid.shape[-2]

        # # Apply interaction layers
        for layer in self.interaction_layers:
            x = layer(x, kernel_basis, fiber_kernel_basis)

        # Readout layer
        readout = self.readout(x)

        # Split scalar and vector parts
        readout_scalar, readout_vec = jnp.split(readout, [self.scalar_num_out], axis=-1)

        # Average over the orientation dimension
        output_scalar = readout_scalar.mean(axis=-2)
        if self.vec_num_out > 0:
            output_vector = jnp.einsum('bnoc,od->bncd', readout_vec, self.ori_grid) / self.ori_grid.shape[-2]
        else:
            output_vector = None

        # Global pooling
        if self.global_pool:
            output_scalar = jnp.sum(output_scalar, axis=1) / output_scalar.shape[1]
            if self.vec_num_out > 0:
                output_vector = jnp.sum(output_vector, axis=1) / output_vector.shape[1]

        return output_scalar, output_vector


class PonitaODE(nn.Module):
    num_hidden: int
    num_layers: int
    scalar_num_out: int
    vec_num_out: int

    invariant: nn.Module
    spatial_dim: int
    num_ori: int
    basis_dim: int
    degree: int
    widening_factor: int
    global_pool: bool
    last_feature_conditioning: bool = False
    kernel_size: Union[float, str] = "global"

    def setup(self):
        self.ponita = PonitaFixedSize(
            num_hidden=self.num_hidden,
            num_layers=self.num_layers,
            scalar_num_out=self.scalar_num_out,
            vec_num_out=self.vec_num_out,
            spatial_dim=self.spatial_dim,
            invariant=self.invariant,
            num_ori=self.num_ori,
            basis_dim=self.basis_dim,
            degree=self.degree,
            widening_factor=self.widening_factor,
            global_pool=self.global_pool,
            last_feature_conditioning=self.last_feature_conditioning,
            kernel_size=self.kernel_size,
        )

    def __call__(self, latents):
        # Unpack latents
        p, a, window = latents

        # a is distributed with 1 mean, so subtract this
        a = a - 1

        # a is scaled with 0.1 std, so divide by this
        # a = a / 0.

        # Get the latent derivative
        try:
            output_scalar, output_vector = self.ponita((p, a, window))
        except Exception as e:
            print(e)
            print("p", p)
            output_scalar, output_vector = self.ponita((p, a, window))
            raise e
        derivative_a = output_scalar
        derivative_p_pos = output_vector.squeeze(2)

        return derivative_p_pos, derivative_a, jnp.zeros_like(window)


if __name__ == "__main__":
    model = PonitaFixedSize(
        num_hidden=64,
        num_layers=3,
        scalar_num_out=1,
        vec_num_out=1,
        spatial_dim=2,
        num_ori=4,
        basis_dim=64,
        degree=3,
        widening_factor=4,
        kernel_size=0.25,
        global_pool=True
    )

    # Create meshgrid
    pos = jnp.stack(jnp.meshgrid(jnp.linspace(-1, 1, 28), jnp.linspace(-1, 1, 28)), axis=-1).reshape(-1, 2)
    # Repeat over batch dim
    pos = jnp.repeat(pos[None, ...], 4, axis=0)
    x = jnp.ones((4, 28*28, 3))

    # Initialize and apply model
    params = model.init(jax.random.PRNGKey(0), (pos, x, None))
    model.apply(params, (pos, x, None))
    print("Success!")
