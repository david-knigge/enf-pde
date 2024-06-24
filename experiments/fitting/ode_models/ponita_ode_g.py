from typing import Union

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


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
        for it in range(1, self.degree + 1):
            polynomial_list.append(jnp.einsum('...i,...j->...ij', polynomial_list[-1], x).reshape(*x.shape[:-1], -1))
        return jnp.concatenate(polynomial_list, axis=-1)


class ConvBlock(nn.Module):
    num_hidden: int
    basis_dim: int
    widening_factor: int

    def setup(self):
        self.conv = SepGconv(self.num_hidden, self.basis_dim)
        self.act_fn = nn.gelu
        self.linear_1 = nn.Dense(self.widening_factor * self.num_hidden)
        self.linear_2 = nn.Dense(self.num_hidden)
        self.norm = nn.LayerNorm()

    def __call__(self, x, kernel_basis):
        x = self.conv(x, kernel_basis)
        x = self.norm(x)
        x = self.linear_1(x)
        x = self.act_fn(x)
        x = self.linear_2(x)
        return x


class SepGconv(nn.Module):
    num_hidden: int
    basis_dim: int
    bias: bool = True

    def setup(self):
        # Set up kernel coefficient layers, one for the spatial kernel and one for the group kernel.
        # This maps from the invariants to a basis for the kernel 2/3->basis_dim.
        self.kernel = nn.Dense(self.num_hidden, use_bias=False, kernel_init=chang_xavier_uniform)

        # Construct bias
        if self.bias:
            self.bias_param = self.param('bias', nn.initializers.zeros, (self.num_hidden,))

    def __call__(self, a, kernel_basis):
        """ Perform separable convolution on fully connected pointcloud.

        Args:
            x: Array of shape (batch, num_points, num_ori, num_features)
            kernel_basis: Array of shape (batch, num_points, num_points, num_ori, basis_dim)
            fiber_kernel_basis: Array of shape (batch, num_points, num_points, basis_dim)
        """
        kernel = self.kernel(kernel_basis)

        # Perform the appearance convolution [batch, senders, channels] * [batch, senders, receivers, channels]
        # -> [batch, receivers, channels]
        a = jnp.einsum('bsc,brsc->brc', a, kernel)

        # Add bias
        if self.bias:
            a = a + self.bias_param
        return a


class PonitaGen(nn.Module):
    num_hidden: int
    num_layers: int
    scalar_num_out: int
    vec_num_out: int

    invariant: nn.Module

    basis_dim: int
    degree: int
    widening_factor: int
    global_pool: bool
    kernel_size: Union[float, str] = "global"

    def setup(self):
        assert self.kernel_size == "global" or self.kernel_size > 0, "kernel_size must be 'global' or a positive number."

        # Initialize kernel basis
        self.kernel_basis = nn.Sequential([
            PolynomialFeatures(degree=self.degree), nn.Dense(self.num_hidden), nn.gelu, nn.Dense(self.basis_dim),
            nn.gelu])

        # Initial node embedding
        self.a_stem = nn.Dense(self.num_hidden, use_bias=False)

        # Make feedforward network
        interaction_layers = []
        for i in range(self.num_layers):
            interaction_layers.append(ConvBlock(
                self.num_hidden,
                self.basis_dim,
                self.widening_factor
            ))
        self.interaction_layers = interaction_layers

        # Readout layers
        self.readout_scalar = nn.Sequential([
            nn.Dense(self.scalar_num_out, use_bias=False, kernel_init=nn.initializers.variance_scaling(1e-6, 'fan_in', 'truncated_normal')),
        ])

        # We average relative positions to other nodes, as well as their orientations, so we need two times the output
        # dimensionality.
        if self.vec_num_out > 0:
            self.readout_vec_rel = nn.Dense(self.vec_num_out, use_bias=False, kernel_init=nn.initializers.variance_scaling(1e-6, 'fan_in', 'truncated_normal'))
            if self.invariant.num_z_ori_dims > 0:
                self.readout_vec_ori = nn.Dense(self.vec_num_out, use_bias=False, kernel_init=nn.initializers.variance_scaling(1e-6, 'fan_in', 'truncated_normal'))
        else:
            self.readout_vec_rel, self.readout_vec_ori = None, None

    def __call__(self, latent):
        """ Forward pass through the network.

        Args:
            pos: Array of shape (batch, num_points, spatial_dim)
            x: Array of shape (batch, num_points, num_in)
        """
        p, a, _ = latent

        # Unpack poses
        if self.invariant.num_z_ori_dims > 0:
            p_pos, p_angles = p[:, :, :self.invariant.num_z_pos_dims], p[:, :, self.invariant.num_z_pos_dims:]
            p = jnp.concatenate((p_pos, jnp.cos(p_angles), jnp.sin(p_angles)), axis=-1)

        # Calculate the invariants
        invariants = self.invariant(p, p)

        # Get kernel
        kernel_basis = self.kernel_basis(invariants)
        if self.kernel_size != "global":
            kernel_basis = (kernel_basis *
                            jnp.exp(-jnp.linalg.norm(p[:, :, None, :] - p[:, None, :, :], axis=-1) / self.kernel_size))

        # Embed the appearance vector
        a = self.a_stem(a)

        # Apply interaction layers
        for layer in self.interaction_layers:
            a = layer(a, kernel_basis)

        # Readout layer, average over all nodes if global
        scalar_out = self.readout_scalar(a)

        if self.vec_num_out > 0:
            # Calculate relative positions to other nodes, this is basis for vector output
            rel_pos = p[:, :, None, :self.invariant.num_z_pos_dims] - p[:, None, :, :self.invariant.num_z_pos_dims]

            # Concatenate a to the invariants
            invariants = jnp.concatenate([
                invariants,
                jnp.broadcast_to(a[:, None, :, :], invariants.shape[:-1] + (a.shape[-1],))], axis=-1)

            readout_vec_rel = self.readout_vec_rel(invariants)
            vec_out = (readout_vec_rel * rel_pos).mean(axis=-2)

            # If the invariant has orientations, use these as additional basis for the vector output
            if self.invariant.num_z_ori_dims > 0:
                p_ori = jnp.broadcast_to(p[:, None, :, self.invariant.num_z_pos_dims:], rel_pos.shape)

                # Calculate the readout vector kernel
                readout_vec_ori = self.readout_vec_ori(invariants)
                vec_out = vec_out + (readout_vec_ori * p_ori).mean(axis=-2)
        else:
            vec_out = None

        if self.global_pool:
            scalar_out = scalar_out.mean(axis=1)
            if vec_out is not None:
                vec_out = vec_out.mean(axis=1)

        return scalar_out, vec_out


class PonitaODEGen(nn.Module):
    num_hidden: int
    num_layers: int
    scalar_num_out: int
    vec_num_out: int

    invariant: nn.Module
    basis_dim: int
    degree: int
    widening_factor: int
    global_pool: bool
    kernel_size: Union[float, str] = "global"

    def setup(self):
        # If we have orientation, we need to add an angle update to the output
        if self.invariant.num_z_ori_dims > 0:
            scalar_num_out = self.scalar_num_out + 1
        else:
            scalar_num_out = self.scalar_num_out

        self.ponita = PonitaGen(
            num_hidden=self.num_hidden,
            num_layers=self.num_layers,
            scalar_num_out=scalar_num_out,
            vec_num_out=self.vec_num_out,
            invariant=self.invariant,
            basis_dim=self.basis_dim,
            degree=self.degree,
            widening_factor=self.widening_factor,
            global_pool=self.global_pool,
            kernel_size=self.kernel_size,
        )

    def __call__(self, latents):
        # Unpack latents
        p, a, window = latents

        # a is distributed with 1 mean, so subtract this
        a = a - 1

        # Get the latent derivative
        output_scalar, output_vector = self.ponita((p, a, window))

        # Optionally perform angle update.
        if self.invariant.num_z_ori_dims > 0:
            derivative_a = output_scalar[:, :, :-1]
            derivative_angle = output_scalar[:, :, -1:]

            derivative_p_pos = output_vector
            derivative_p = jnp.concatenate([derivative_p_pos, derivative_angle], axis=-1)
        else:
            derivative_a = output_scalar
            derivative_p = output_vector

        if window is not None:
            window_der = jnp.zeros_like(window)
        else:
            window_der = None

        return derivative_p, derivative_a, window_der  # No derivative for window


if __name__ == "__main__":
    from enf.steerable_attention.invariant import get_sa_invariant

    class Cfg:
        num_in = 2
        invariant_type = "rel_pos"

    # Init invariant
    self_attn_invariant = get_sa_invariant(Cfg())

    # ponita
    ode_model = PonitaGen(
        num_hidden=128,
        num_layers=3,
        scalar_num_out=8,
        vec_num_out=1,
        invariant=self_attn_invariant,
        basis_dim=32,
        degree=3,
        widening_factor=2,
        kernel_size="global",
        global_pool=False,
    )

    p_pos = jnp.array([[[-0.5, -0.5], [-0.3, 0.7], [0.1, -0.1], [0.2, 0.2]]])
    p_angles = jnp.array([[[0.0], [-0.75], [1.0], [1.5]]])
    p = jnp.concatenate([p_pos, p_angles], axis=-1)

    a = jnp.ones((1, 4, 8))

    # Initialize model
    latents = (p, a, jnp.zeros((1, 4, 1)))
    ode_params = ode_model.init(jax.random.PRNGKey(0), latents)

    # Plot the appearance
    import matplotlib.pyplot as plt

    # Plot the orientation vectors at every 5th point
    p_ori = jnp.concatenate([jnp.cos(p[:, :, 2:]), jnp.sin(p[:, :, 2:])], axis=-1)
    p_pos = p[:, :, :2]
    plt.quiver(p_pos[0, :, 0], p_pos[0, :, 1], p_ori[0, :, 0], p_ori[0, :, 1], color='r')

    # Set x and y lims
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    # Forward pass
    derivative_a, derivative_p_pos = ode_model.apply(ode_params, latents)
    print("derivative a:", derivative_a.mean(axis=-1))

    # Apply updates
    new_p = p.at[:, :, :2].add(derivative_p_pos)

    p_ori = jnp.concatenate([jnp.cos(p[:, :, 2:]), jnp.sin(p[:, :, 2:])], axis=-1)
    p_pos = new_p[:, :, :2]
    plt.quiver(p_pos[0, :, 0], p_pos[0, :, 1], p_ori[0, :, 0], p_ori[0, :, 1], color='b')
    plt.show()

    # Now we rotate the orientations and positions with 45 degrees
    p_angles = p[:, :, 2:]
    p_pos = p[:, :, :2]
    p = jnp.concatenate([p_pos, p_ori], axis=-1)

    # Create rotation matrix
    rot_matrix = jnp.array([[jnp.cos(jnp.pi/4), -jnp.sin(jnp.pi/4)], [jnp.sin(jnp.pi/4), jnp.cos(jnp.pi/4)]])
    p_angles = p_angles - jnp.pi/4
    # Rotate
    rot_p_pos = jnp.einsum('ijk,kl->ijl', p[:, :, :2], rot_matrix)
    rot_p = jnp.concatenate([rot_p_pos, p_angles], axis=-1)

    plt.quiver(rot_p[0, :, 0], rot_p[0, :, 1], jnp.cos(rot_p[0, :, 2]), jnp.sin(rot_p[0, :, 2]), color='r')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    # Forward
    latents = (rot_p, a, jnp.zeros((1, 4, 1)))
    derivative_a, derivative_p_pos = ode_model.apply(ode_params, latents)
    print("derivative a:", derivative_a.mean(axis=-1))
    # Update
    new_p = rot_p.at[:, :, :2].add(derivative_p_pos)
    # appear = appear + derivative_a

    p_ori = jnp.concatenate([jnp.cos(rot_p[:, :, 2:]), jnp.sin(rot_p[:, :, 2:])], axis=-1)
    p_pos = new_p[:, :, :2]
    plt.quiver(p_pos[0, :, 0], p_pos[0, :, 1], p_ori[0, :, 0], p_ori[0, :, 1], color='b')

    plt.show()
