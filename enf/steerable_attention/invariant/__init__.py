from enf.steerable_attention.invariant._base_invariant import BaseInvariant
from enf.steerable_attention.invariant.norm_rel_pos import NormRelativePositionND
from enf.steerable_attention.invariant.rel_pos import RelativePositionND
from enf.steerable_attention.invariant.ponita import Ponita2D, PonitaPos2D
from enf.steerable_attention.invariant.abs_pos import AbsolutePositionND
from enf.steerable_attention.invariant.rel_pos_periodic import RelativePosition2DPeriodic
from enf.steerable_attention.invariant.polar_periodic import RelativePositionPolarPeriodic
from enf.steerable_attention.invariant.spherical_longitude import RelativeLatitudePeriodic
from enf.steerable_attention.invariant.ball import BallInvariant
from enf.steerable_attention.invariant.ball_lat import BallLatInvariant


def get_sa_invariant(cfg) -> BaseInvariant:
    """ Get the invariant for the self attention module.

        Args:
            name (str): The name of the invariant.

        Returns:
            BaseInvariant: The invariant module.

        """
    if cfg.invariant_type == "norm_rel_pos":
        return NormRelativePositionND(num_dims=cfg.num_in)
    elif cfg.invariant_type == "rel_pos":
        return RelativePositionND(num_dims=cfg.num_in)
    elif cfg.invariant_type == "rel_pos_periodic":
        assert cfg.num_in == 2, "RelativePosition2DPeriodic currently only supports 2D input."
        return RelativePosition2DPeriodic(num_dims=cfg.num_in)
    elif cfg.invariant_type == "ponita":
        assert cfg.num_in == 2, "Ponita2D currently only supports 2D input."
        return Ponita2D()
    elif cfg.invariant_type == "abs_pos":
        return AbsolutePositionND(num_dims=cfg.num_in)
    elif cfg.invariant_type == "polar_periodic":
        return RelativePositionPolarPeriodic()
    elif cfg.invariant_type == "latitude_periodic":
        return RelativeLatitudePeriodic()
    elif cfg.invariant_type == "ball":
        return BallInvariant()
    elif cfg.invariant_type == "ball_lat":
        return BallLatInvariant()
    else:
        raise ValueError(f"Unknown invariant type: {cfg.invariant_type}.")


def get_ca_invariant(cfg) -> BaseInvariant:
    """ Get the invariant for the cross attention module.

    Args:
        name (str): The name of the invariant.

    Returns:
        BaseInvariant: The invariant module.

    """
    if cfg.invariant_type == "norm_rel_pos":
        return NormRelativePositionND(num_dims=cfg.num_in)
    elif cfg.invariant_type == "rel_pos":
        return RelativePositionND(num_dims=cfg.num_in)
    elif cfg.invariant_type == "rel_pos_periodic":
        assert cfg.num_in == 2, "RelativePosition2DPeriodic currently only supports 2D input."
        return RelativePosition2DPeriodic(num_dims=cfg.num_in)
    elif cfg.invariant_type == "ponita":
        assert cfg.num_in == 2, "Ponita2D currently only supports 2D input."
        return PonitaPos2D()
    elif cfg.invariant_type == "abs_pos":
        return AbsolutePositionND(num_dims=cfg.num_in)
    elif cfg.invariant_type == "polar_periodic":
        return RelativePositionPolarPeriodic()
    elif cfg.invariant_type == "latitude_periodic":
        return RelativeLatitudePeriodic()
    elif cfg.invariant_type == "ball":
        return BallInvariant()
    elif cfg.invariant_type == "ball_lat":
        return BallLatInvariant()
    else:
        raise ValueError(f"Unknown invariant type: {cfg.invariant_type}.")
