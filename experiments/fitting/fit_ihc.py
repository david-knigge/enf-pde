import hydra
from omegaconf import DictConfig
import omegaconf
import wandb

import jax.numpy as jnp

from experiments.fitting.datasets import get_dataloader
from experiments.fitting.trainers.pde_trainer import MetaSGDPDETrainer

from experiments.fitting import get_model_pde


@hydra.main(version_base=None, config_path=".", config_name="config_ihc")
def train(cfg: DictConfig):

    # Set log dir
    if not cfg.logging.log_dir:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        cfg.logging.log_dir = hydra_cfg['runtime']['output_dir']

    assert cfg.dataset.name == "ihc"

    # Create the dataset, this is a low-res version of shallow-water
    trainset, testset = get_dataloader(dataset_cfg=cfg.dataset)

    sample_batch = next(iter(trainset))
    smp_image = sample_batch[0][0]
    image_shape = smp_image.shape
    cfg.dataset.image_shape = image_shape

    # coordinate grid over sphere
    phi_grid = jnp.linspace(0, 2 * jnp.pi, 48, endpoint=False)
    theta_grid = jnp.linspace(0+1e-3, jnp.pi, 24, endpoint=False)
    r_grid = jnp.linspace(0, 1, 24)
    phi, theta, r = jnp.meshgrid(phi_grid, theta_grid, r_grid, indexing='ij')
    coords = jnp.stack([phi, theta, r], axis=-1).reshape(-1, 3) #[ phi, theta, r]

    # Set dimensionality of input and output
    cfg.nef.num_in = 3
    cfg.nef.num_out = image_shape[-1]

    # Initialize wandb
    wandb.init(
        project=cfg.proj_name,
        name=f"meta_{cfg.dataset.name}_ad",
        dir=cfg.logging.log_dir,
        config=omegaconf.OmegaConf.to_container(cfg),
        mode='disabled' if cfg.logging.debug else 'online',
    )

    # Get nef and autodecoders
    nef, ode_model = get_model_pde(cfg)

    trainer = MetaSGDPDETrainer(
        nef=nef,
        ode_model=ode_model,
        config=cfg,
        train_loader=trainset,
        val_loader=testset,
        coords=coords,
        seed=cfg.seed,
    )
    trainer.create_functions()

    if cfg.logging.load_from_checkpoint:
        train_state = trainer.load_checkpoint()
    else:
        train_state = None

    # Train model
    final_state = trainer.train_model(cfg.training.num_epochs, state=train_state)


if __name__ == "__main__":
    train()
