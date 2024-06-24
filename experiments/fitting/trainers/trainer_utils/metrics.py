import jax
import jax.numpy as jnp
import numpy as np


def iou(occ1: jax.Array, occ2: jax.Array) -> np.ndarray:
    """Computes the Intersection over Union (IoU) value for two sets of occupancy values.

    NOTE: ASSUMES THAT OCCUPANCY VALUES ARE IN THE RANGE [-inf, inf].

    The formula used is the following:

    .. math::
        \\text{IoU} = \\frac{|A \\cap B|}{|A \\cup B|}

    :param occ1: first set of occupancy values
    :type occ1: jax.Array
    :param occ2: second set of occupancy values
    :type occ2: jax.Array

    :return: IoU value
    :rtype: np.ndarray
    """
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = occ1 >= 0.0
    occ2 = occ2 >= 0.0

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = area_intersect / area_union

    return iou


def psnr(
    image: jnp.ndarray, ground_truth: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray
) -> jnp.ndarray:
    """Computes the Peak Signal to Noise Ration (PSNR). Peak signal found from ground_truth and
    noise is given as the mean square error (MSE) between image and ground_truth. The value is
    returned in decibels.

    https://github.com/photosynthesis-team/piq/blob/master/piq/psnr.py

    :param image: the first tensor used in the calculation. [batch, height, width, channels]
    :type image: jnp.ndarray
    :param ground_truth: the second tensor used in the calculation. [batch, height, width, channels]
    :type ground_truth: jnp.ndarray
    :param mean: the mean of the dataset.
    :type mean: jnp.ndarray
    :param std: the standard deviation of the dataset.
    :type std: jnp.ndarray

    :return: the mean of the PNSR of the image according to the peak signal that can be obtained in ground_truth:
        mean(log10(peak_signal**2/MSE(image-ground_truth))).
    :rtype: jnp.ndarray
    """
    # change the view to compute the metric in a batched way correctly
    w_image = image# * std + mean
    w_ground_truth = ground_truth# * std + mean

    maxval = jnp.max(w_ground_truth)

    w_image = w_image / maxval
    w_ground_truth = w_ground_truth / maxval

    EPS = 1e-8

    mse = jnp.maximum(0, jnp.mean((w_image - w_ground_truth) ** 2, axis=(-1, -2, -3)))

    return -10 * jnp.log10(mse + EPS)


def mse(a: jax.Array, b: jax.Array) -> jax.Array:
    """Returns the Mean Squared Error between `a` and `b`.

    :param a: First image (or set of images).
    :type a: jax.Array
    :param b: Second image (or set of images).
    :type b: jax.Array

    :return: MSE between `a` and `b`.
    :rtype: jax.Array
    """
    return jnp.square(a - b).mean()
