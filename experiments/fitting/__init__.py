import math

from functools import partial

from enf.models import EquivariantCrossAttentionNeF
from enf.steerable_attention.invariant import get_sa_invariant, get_ca_invariant
from enf.latents.autodecoder import PositionOrientationFeatureAutodecoder
from enf.latents.autodecoder_meta import PositionOrientationFeatureAutodecoderMeta
from experiments.fitting.ode_models.ponita_ode import PonitaODE
from experiments.fitting.ode_models.mlp_ode import MLPODE
from experiments.fitting.ode_models.ponita_ode_g import PonitaODEGen


def get_model_pde(cfg):
    """ Get autodecoders and enf based on the configuration. """

    # Init invariant
    self_attn_invariant = get_sa_invariant(cfg.nef)
    cross_attn_invariant = get_ca_invariant(cfg.nef)

    # Calculate initial gaussian window size
    assert math.sqrt(cfg.nef.num_latents)

    # Init model
    nef = EquivariantCrossAttentionNeF(
        num_hidden=cfg.nef.num_hidden,
        num_heads=cfg.nef.num_heads,
        num_layers=cfg.nef.num_layers,
        num_out=cfg.nef.num_out,
        latent_dim=cfg.nef.latent_dim,
        self_attn_invariant=self_attn_invariant,
        cross_attn_invariant=cross_attn_invariant,
        embedding_type=cfg.nef.embedding_type,
        embedding_freq_multiplier=[cfg.nef.embedding_freq_multiplier_invariant,
                                   cfg.nef.embedding_freq_multiplier_value],
        condition_value_transform=cfg.nef.condition_value_transform,
        use_gaussian_window=cfg.nef.use_gaussian_window,
    )

    # Select ode model
    if cfg.node.name == "mlp":
        ode_model = MLPODE(
            num_hidden=cfg.node.num_hidden,
            num_layers=cfg.node.num_layers,
            scalar_num_out=cfg.nef.latent_dim,
            vec_num_out=1,
        )
    elif cfg.node.name == "ponita":
        # ponita
        ode_model = PonitaODEGen(
            num_hidden=cfg.node.num_hidden,
            num_layers=cfg.node.num_layers,
            scalar_num_out=cfg.nef.latent_dim,
            invariant=self_attn_invariant,
            vec_num_out=1,
            basis_dim=cfg.node.basis_dim,
            degree=cfg.node.degree,
            widening_factor=cfg.node.widening_factor,
            kernel_size="global",
            global_pool=False,
        )
    else:
        raise ValueError(f"Unknown ODE model: {cfg.node.name}")

    return nef, ode_model
