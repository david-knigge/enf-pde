import jax.numpy as jnp

from enf.steerable_attention.invariant._base_invariant import BaseInvariant


class NormRelativePositionND(BaseInvariant):

    def __init__(self, num_dims: int):
        """ Calculate the relative position between two sets of coordinates in N dimensions.

        Args:
            num_dims (int): The dimensionality of the coordinates, corresponds to the dimensionality of the translation
                group.
        """
        super().__init__()
        
        # Set the dimensionality of the invariant.
        self.dim = 1
        self.num_x_pos_dims = num_dims
        self.num_x_ori_dims = 0
        self.num_z_pos_dims = num_dims
        self.num_z_ori_dims = 0

    def __call__(self, x, p):
        """ Calculate the relative position between two sets of coordinates.

        Args:
            x (torch.Tensor): The pose of the input coordinates. Shape (batch_size, num_coords, num_x_pos_dims).
            p (torch.Tensor): The pose of the latent points. Shape (batch_size, num_latents, num_z_pos_dims).
        Returns:
            invariants (torch.Tensor): The relative position between the input and latent coordinates.
                Shape (batch_size, num_coords, num_latents, 1).
        """
        return jnp.linalg.norm(p[:, None, :, :] - x[:, :, None, :], ord=2, axis=-1, keepdims=True)
