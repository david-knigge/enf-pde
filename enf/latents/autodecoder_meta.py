import jax.numpy as jnp

from enf.latents.autodecoder import PositionOrientationFeatureAutodecoder


class PositionOrientationFeatureAutodecoderMeta(PositionOrientationFeatureAutodecoder):

    def __call__(self):
        # Implement the forward pass using JAX operations
        p_pos = self.p_pos

        if self.num_ori_dims > 0:
            p_ori = self.p_ori
            p = jnp.concatenate((p_pos, p_ori), axis=-1)
        else:
            p = p_pos

        a = self.a

        # Optionally, get the gaussian window for the latent points
        if self.gaussian_window_size is not None:
            gaussian_window = self.gaussian_window
        else:
            gaussian_window = None
        return p, a, gaussian_window
