import hydra
from omegaconf import DictConfig
import omegaconf
import wandb

import jax.numpy as jnp

from experiments.fitting.datasets import get_dataloader
from experiments.fitting.trainers.pde_trainer import MetaSGDPDETrainer

from experiments.fitting import get_model_pde


@hydra.main(version_base=None, config_path=".", config_name="config_navier_stokes")
def generate_data(cfg: DictConfig):

    # Set log dir
    if not cfg.logging.log_dir:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        cfg.logging.log_dir = hydra_cfg['runtime']['output_dir']

    # Create the dataset
    trainset, testset = get_dataloader(dataset_cfg=cfg.dataset)

    # Loop over trainset
    for i, batch in enumerate(trainset):
        pass

    # Loop over testset
    for i, batch in enumerate(testset):
        pass


if __name__ == "__main__":
    generate_data()
