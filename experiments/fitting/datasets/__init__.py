from experiments.fitting.datasets.pdes import NavierStokesDataset, DiffusionSphereDataset, ShallowWaterDataset, CahnHilliardDataset, DiffusionDataset, ShallowWaterRandInit, ShallowWaterRandInitHalfRes, InternallyHeatedConvection

from typing import Union, Any, Sequence

import numpy as np
from torch.utils import data
import torchvision
import torch
import math
import shelve
import os


def image_to_numpy(image):
    return np.array(image) / 255


def add_channel_axis(image: np.ndarray):
    return image[..., np.newaxis]


def permute_image_channels(image: np.ndarray):
    if len(image.shape) == 3:
        return np.moveaxis(image, 2, 0)
    else:
        return image


def numpy_collate(batch: Union[np.ndarray, Sequence[Any], Any]):
    """
    TODO: this might be a repeat, maybe it's ok to make it special for shapes, but needs a check
    Collate function for numpy arrays.

    This function acts as replacement to the standard PyTorch-tensor collate function in PyTorch DataLoader.

    Args:
        batch: Batch of data. Can be a numpy array, a list of numpy arrays, or nested lists of numpy arrays.

    Returns:
        Batch of data as (potential list or tuple of) numpy array(s).
    """
    if isinstance(batch, np.ndarray):
        return batch
    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def ns_data_transform(batch: Union[np.ndarray, Sequence[Any], Any]):
    batch = numpy_collate(batch)
    data, coords, index = batch
    data = np.concatenate(data, axis=0)
    return data, coords, index



def get_dataloader(dataset_cfg):

    if dataset_cfg.name == "navier_stokes":

        n_frames_train = 20
        t_horizon = 20
        size = 64
        n_seq = dataset_cfg.num_signals_train
        tt = torch.linspace(0, 1, size + 1)[0:-1]
        X, Y = torch.meshgrid(tt, tt)
        visc = 1e-3
        dataset_tr_params = {
            "device": "cuda", "n_seq": n_seq, "n_seq_per_traj": 1, "t_horizon": t_horizon, "dt": 1,
            "size": size,
            "group": "train", 'n_frames_train': n_frames_train,
            "param": {"f": 0.3 * (torch.cos(4 * math.pi * X) + torch.cos(4 * math.pi * Y)), "visc": visc}
        }

        dataset_tr_eval_params = dict()
        dataset_tr_eval_params.update(dataset_tr_params)
        dataset_tr_eval_params["group"] = "train_eval"

        dataset_ts_params = dict()
        dataset_ts_params.update(dataset_tr_params)
        dataset_ts_params["group"] = "test"
        dataset_ts_params["n_seq"] = dataset_cfg.num_signals_test

        os.makedirs(dataset_cfg.path + 'unroll-ns-simple', exist_ok=True)
        buffer_file_tr = dataset_cfg.path + 'unroll-ns-simple/shelve_train.shelve'
        buffer_file_ts = dataset_cfg.path + 'unroll-ns-simple/shelve_test.shelve'

        buffer_shelve_tr = shelve.open(buffer_file_tr)
        buffer_shelve_ts = shelve.open(buffer_file_ts)
        train_dset = NavierStokesDataset(buffer_shelve=buffer_shelve_tr, **dataset_tr_params, normalize=False)
        test_dset = NavierStokesDataset(buffer_shelve=buffer_shelve_ts, **dataset_ts_params, normalize=False)

    elif dataset_cfg.name == "navier_stokes_long":

        n_frames_train = dataset_cfg.traj_len_train + dataset_cfg.traj_len_out_horizon
        t_horizon = dataset_cfg.traj_len_train + dataset_cfg.traj_len_out_horizon
        size = 64
        n_seq = dataset_cfg.num_signals_train
        tt = torch.linspace(0, 1, size + 1)[0:-1]
        X, Y = torch.meshgrid(tt, tt)
        visc = 1e-3
        dataset_tr_params = {
            "device": "cuda", "n_seq": n_seq, "n_seq_per_traj": 1, "t_horizon": t_horizon, "dt": 1,
            "size": size,
            "group": "train", 'n_frames_train': n_frames_train,
            "param": {"f": 0.3 * (torch.cos(4 * math.pi * X) + torch.cos(4 * math.pi * Y)), "visc": visc}
        }

        dataset_tr_eval_params = dict()
        dataset_tr_eval_params.update(dataset_tr_params)
        dataset_tr_eval_params["group"] = "train_eval"

        dataset_ts_params = dict()
        dataset_ts_params.update(dataset_tr_params)
        dataset_ts_params["group"] = "test"
        dataset_ts_params["n_seq"] = dataset_cfg.num_signals_test

        os.makedirs(dataset_cfg.path + 'navier_stokes_long', exist_ok=True)
        buffer_file_tr = dataset_cfg.path + 'navier_stokes_long/shelve_train.shelve'
        buffer_file_ts = dataset_cfg.path + 'navier_stokes_long/shelve_test.shelve'

        buffer_shelve_tr = shelve.open(buffer_file_tr)
        buffer_shelve_ts = shelve.open(buffer_file_ts)
        print(buffer_file_ts, list(buffer_shelve_ts.keys()))
        train_dset = NavierStokesDataset(buffer_shelve=buffer_shelve_tr, **dataset_tr_params, normalize=False)
        test_dset = NavierStokesDataset(buffer_shelve=buffer_shelve_ts, **dataset_ts_params, normalize=False)

    elif dataset_cfg.name == "diff_sphere":

        n_frames_train = 20
        size = (128, 64)
        n_seq = 512
        t_horizon = 20

        dataset_tr_params = {
            "device": "cuda", "n_seq": n_seq, "n_seq_per_traj": 1, "t_horizon": t_horizon, "dt": 0.5,
            "size": size,
            "group": "train", 'n_frames_train': n_frames_train,
            "param": None
        }

        dataset_tr_eval_params = dict()
        dataset_tr_eval_params.update(dataset_tr_params)
        dataset_tr_eval_params["group"] = "train_eval"

        dataset_ts_params = dict()
        dataset_ts_params.update(dataset_tr_params)
        dataset_ts_params["group"] = "test"
        dataset_ts_params["n_seq"] = 128

        os.makedirs(dataset_cfg.path + 'diffsphere', exist_ok=True)
        buffer_file_tr = dataset_cfg.path + 'diffsphere/shelve_train.shelve'
        buffer_file_ts = dataset_cfg.path + 'diffsphere/shelve_test.shelve'

        buffer_shelve_tr = shelve.open(buffer_file_tr)
        buffer_shelve_ts = shelve.open(buffer_file_ts)
        train_dset = DiffusionSphereDataset(buffer_shelve=buffer_shelve_tr, **dataset_tr_params, normalize=False)
        test_dset = DiffusionSphereDataset(buffer_shelve=buffer_shelve_ts, **dataset_ts_params, normalize=False)

    elif dataset_cfg.name == "shallow_water":

        n_frames_train = 20
        size = (192, 96)
        n_seq = 512
        t_horizon = 20

        dataset_tr_params = {
            "device": "cuda", "n_seq": n_seq, "n_seq_per_traj": 1, "t_horizon": t_horizon, "dt": 0.5,
            "size": size,
            "group": "train", 'n_frames_train': n_frames_train,
            "param": None
        }

        dataset_tr_eval_params = dict()
        dataset_tr_eval_params.update(dataset_tr_params)
        dataset_tr_eval_params["group"] = "train"

        dataset_ts_params = dict()
        dataset_ts_params.update(dataset_tr_params)
        dataset_ts_params["group"] = "test"
        dataset_ts_params["n_seq"] = dataset_cfg.num_signals_test


        os.makedirs(dataset_cfg.path + '.sw-vorticity', exist_ok=True)
        buffer_file_tr = dataset_cfg.path + 'sw-vorticity/shelve_train.shelve'
        buffer_file_ts = dataset_cfg.path + 'sw-vorticity/shelve_test.shelve'

        buffer_shelve_tr = shelve.open(buffer_file_tr)
        buffer_shelve_ts = shelve.open(buffer_file_ts)
        train_dset = ShallowWaterRandInit(buffer_shelve=buffer_shelve_tr, **dataset_tr_params, normalize=False)
        test_dset = ShallowWaterRandInit(buffer_shelve=buffer_shelve_ts, **dataset_ts_params, normalize=False)

    elif dataset_cfg.name == "shallow_water_low_res":

        n_frames_train = 20
        size = (192, 96)
        n_seq = 512
        t_horizon = 20

        dataset_tr_params = {
            "device": "cuda", "n_seq": n_seq, "n_seq_per_traj": 1, "t_horizon": t_horizon, "dt": 0.5,
            "size": size,
            "group": "train", 'n_frames_train': n_frames_train,
            "param": None
        }

        dataset_tr_eval_params = dict()
        dataset_tr_eval_params.update(dataset_tr_params)
        dataset_tr_eval_params["group"] = "train"

        dataset_ts_params = dict()
        dataset_ts_params.update(dataset_tr_params)
        dataset_ts_params["group"] = "test"
        dataset_ts_params["n_seq"] = dataset_cfg.num_signals_test

        os.makedirs(dataset_cfg.path + 'sw-vorticity', exist_ok=True)
        buffer_file_tr = dataset_cfg.path + 'sw-vorticity/shelve_train.shelve'
        buffer_file_ts = dataset_cfg.path + 'sw-vorticity/shelve_test.shelve'

        buffer_shelve_tr = shelve.open(buffer_file_tr)
        buffer_shelve_ts = shelve.open(buffer_file_ts)
        train_dset = ShallowWaterRandInitHalfRes(buffer_shelve=buffer_shelve_tr, **dataset_tr_params, normalize=False)
        test_dset = ShallowWaterRandInitHalfRes(buffer_shelve=buffer_shelve_ts, **dataset_ts_params, normalize=False)

    elif dataset_cfg.name == "cahn_hilliard":

        n_frames_train = 20
        size = 64
        dt = 20.0
        t_horizon = int(20 * dt) + 9 * dt # we discard 10 frames to avoid boundary effects
        dataset_tr_params = {
            "n_seq": dataset_cfg.num_signals_train, "n_seq_per_traj": 1, "t_horizon": t_horizon, "dt": dt, "size": size, "group": "train",
            'n_frames_train': n_frames_train, "param": {'bc_c': 'periodic', 'bc_mu': 'periodic'}}
        dataset_ts_params = dict()
        dataset_ts_params.update(dataset_tr_params)
        dataset_ts_params["group"] = "test"

        os.makedirs(dataset_cfg.path + 'cahn_hilliard', exist_ok=True)
        buffer_file_tr = dataset_cfg.path + 'cahn_hilliard' + '/shelve_train.shelve'
        buffer_file_ts = dataset_cfg.path + 'cahn_hilliard' + '/shelve_test.shelve'

        buffer_shelve_tr = shelve.open(buffer_file_tr)
        buffer_shelve_ts = shelve.open(buffer_file_ts)
        dataset_ts_params["n_seq"] = dataset_cfg.num_signals_test

        train_dset = CahnHilliardDataset(buffer_shelve=buffer_shelve_tr, **dataset_tr_params)
        test_dset = CahnHilliardDataset(buffer_shelve=buffer_shelve_ts, **dataset_ts_params)

    elif dataset_cfg.name == "diffusion_plane":

        n_frames_train = 20
        size = 64
        dataset_tr_params = {
            "n_seq": dataset_cfg.num_signals_train, "n_seq_per_traj": 1, "t_horizon": 10, "dt": 0.5, "size": size, "group": "train",
            'n_frames_train': n_frames_train, "param": {}}
        dataset_ts_params = dict()
        dataset_ts_params.update(dataset_tr_params)
        dataset_ts_params["group"] = "test"

        os.makedirs(dataset_cfg.path + 'diffusion', exist_ok=True)
        buffer_file_tr = dataset_cfg.path + 'diffusion' + '/shelve_train.shelve'
        buffer_file_ts = dataset_cfg.path + 'diffusion' + '/shelve_test.shelve'

        buffer_shelve_tr = shelve.open(buffer_file_tr)
        buffer_shelve_ts = shelve.open(buffer_file_ts)
        dataset_ts_params["n_seq"] = dataset_cfg.num_signals_test

        train_dset = DiffusionDataset(buffer_shelve=buffer_shelve_tr, **dataset_tr_params)
        test_dset = DiffusionDataset(buffer_shelve=buffer_shelve_ts, **dataset_ts_params)

    elif dataset_cfg.name == "ihc":

        n_frames_train = 20
        size = (48, 24, 24)
        t_horizon = 20

        dataset_tr_params = {
            "n_seq": dataset_cfg.num_signals_train, "n_seq_per_traj": 1, "t_horizon": t_horizon, "dt": 1, "size": size,
            "group": "train",
            'n_frames_train': n_frames_train}

        dataset_tr_eval_params = dict()
        dataset_tr_eval_params.update(dataset_tr_params)
        dataset_tr_eval_params["group"] = "train"

        dataset_ts_params = dict()
        dataset_ts_params.update(dataset_tr_params)
        dataset_ts_params["group"] = "test"
        dataset_ts_params["n_seq"] = dataset_cfg.num_signals_test

        os.makedirs(dataset_cfg.path + 'ihc', exist_ok=True)
        buffer_file_tr = dataset_cfg.path + 'ihc/shelve_train.shelve'
        buffer_file_ts = dataset_cfg.path + 'ihc/shelve_test.shelve'

        buffer_shelve_tr = shelve.open(buffer_file_tr)
        buffer_shelve_ts = shelve.open(buffer_file_ts)

        train_dset = InternallyHeatedConvection(buffer_shelve=buffer_shelve_tr, **dataset_tr_params, normalize=False)
        test_dset = InternallyHeatedConvection(buffer_shelve=buffer_shelve_ts, **dataset_ts_params, normalize=False)

    else:
        raise ValueError(f"Unknown dataset name: {dataset_cfg.name}")

    if dataset_cfg.num_signals_train != -1:
        train_dset = data.Subset(train_dset, np.arange(0, dataset_cfg.num_signals_train))
    if dataset_cfg.num_signals_test != -1:
        test_dset = data.Subset(test_dset, np.arange(0, dataset_cfg.num_signals_test))

    batch_size = min(len(train_dset), dataset_cfg.batch_size)

    train_loader = data.DataLoader(
        train_dset,
        batch_size=batch_size,
        shuffle=True,   # shuffle True doesn't work with navier-stokes
        num_workers=dataset_cfg.num_workers,
        collate_fn=numpy_collate,  # ns_data_transform
        persistent_workers=False,
        drop_last=True
    )

    test_loader = data.DataLoader(
        test_dset,
        batch_size=batch_size,
        shuffle=False,    # shuffle True doesn't work with navier-stokes
        num_workers=dataset_cfg.num_workers,
        collate_fn=numpy_collate,  # ns_data_transform
        persistent_workers=False,
        drop_last=True
    )

    return train_loader, test_loader
