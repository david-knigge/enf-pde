# Copyright 2022 Yuan Yin & Matthieu Kirchmeyer

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from scipy.integrate import odeint
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from netCDF4 import Dataset as netCDFDataset
from pde import ScalarField, CartesianGrid, UnitGrid, MemoryStorage, PDE
from pde.pdes import WavePDE, CahnHilliardPDE, DiffusionPDE
import dedalus.public as d3
import h5py
import math
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)

def get_mgrid(sidelen, vmin=-1, vmax=1, dim=2):
    """
    Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int
    """
    if isinstance(sidelen, int):
        tensors = tuple(dim * [torch.linspace(vmin, vmax, steps=sidelen)])
    elif isinstance(sidelen, (list, tuple)):
        if isinstance(vmin, (list, tuple)) and isinstance(vmax, (list, tuple)):
            tensors = tuple([torch.linspace(mi, ma, steps=l) for mi, ma, l in zip(vmin, vmax, sidelen)])
        else:
            tensors = tuple([torch.linspace(vmin, vmax, steps=l) for l in sidelen])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    return mgrid


def get_mgrid_from_tensors(tensors):
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    return mgrid


class AbstractDataset(Dataset):
    def __init__(self, n_seq, n_seq_per_traj, size, t_horizon, dt, n_frames_train, buffer_shelve, group, scale=1, *args, **kwargs):
        super().__init__()
        self.n_seq = n_seq
        self.n_seq_per_traj = n_seq_per_traj
        self.size = size  # size of the 2D grid
        self.t_horizon = float(t_horizon)  # total time
        self.n = int(t_horizon / dt)  # number of iterations
        self.dt_eval = float(dt)
        assert group in ['train', 'train_eval', 'test', 'test_hr']
        self.group = group
        self.max = np.iinfo(np.int32).max
        self.buffer = dict()
        self.buffer_shelve = buffer_shelve
        self.n_frames_train = n_frames_train
        self.scale = scale

        # Normalize the dataset optionally. Done in the child classes.
        self.normalize = False
        self.min_value, self.max_value, self.mean_value, self.std_value = None, None, None, None

    def calc_min_max_mean_std_values(self):
        # Loop over the dataset to calculate min, max, mean and std values.
        samples = [self[i][0] for i in range(len(self))]
        mean_value = torch.mean(torch.cat(samples, dim=0)).item()
        std_value = torch.std(torch.cat(samples, dim=0)).item()

        min_value = torch.min(self[0][0]).item()
        max_value = torch.max(self[0][0]).item()
        for i in range(1, len(self)):
            min_value = min(min_value, torch.min(self[i][0]).item())
            max_value = max(max_value, torch.max(self[i][0]).item())
        return min_value, max_value, mean_value, std_value

    def _get_init_cond(self, index):
        raise NotImplementedError

    def _generate_trajectory(self, traj_id):
        raise NotImplementedError
    
    def _load_trajectory(self, traj_id):
        raise NotImplementedError

    def __getitem__(self, index):
        t = torch.arange(0, self.t_horizon, self.dt_eval).float()
        traj_id = index // self.n_seq_per_traj
        seq_id = index % self.n_seq_per_traj
        if self.buffer.get(f'{traj_id}') is None:
            if self.buffer_shelve is not None:
                if self.buffer_shelve.get(f'{traj_id}') is None:
                    self._generate_trajectory(traj_id)
                self.buffer[f'{traj_id}'] = self.buffer_shelve[f'{traj_id}']
            else:
                self.buffer[f'{traj_id}'] = self._load_trajectory(traj_id)
        data = self.buffer[f'{traj_id}']['data'][:, seq_id * self.n:(seq_id + 1) * self.n]  # (n_ch, T, H, W)

        # In case of IHC, the data has depth dim
        if len(data.shape) == 5:
            data = np.transpose(data, (1, 2, 3, 4, 0))  # (T, H, W, D, n_ch)
        else:
            data = np.transpose(data, (1, 2, 3, 0))  # (T, H, W, n_ch)

        # Limit the number of frames for training.
        if self.group == 'train':
            data = data[:self.n_frames_train]

        return data, self.coords, index

    def __len__(self):
        return self.n_seq


#################
# Navier Stokes #
#################


class GaussianRF(object):
    def __init__(self, dim, size, alpha=2, tau=3, sigma=None):
        self.dim = dim
        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - self.dim))
        k_max = size // 2
        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1), torch.arange(start=-k_max, end=0, step=1)), 0)
            self.sqrt_eig = size * math.sqrt(2.0) * sigma * ((4 * (math.pi ** 2) * (k ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0] = 0.
        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1),
                                    torch.arange(start=-k_max, end=0, step=1)), 0).repeat(size, 1)
            k_x = wavenumers.transpose(0, 1)
            k_y = wavenumers
            self.sqrt_eig = (size ** 2) * math.sqrt(2.0) * sigma * (
                        (4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0, 0] = 0.0
        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1),
                                    torch.arange(start=-k_max, end=0, step=1)), 0).repeat(size, size, 1)
            k_x = wavenumers.transpose(1, 2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0, 2)
            self.sqrt_eig = (size ** 3) * math.sqrt(2.0) * sigma * (
                        (4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2 + k_z ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0, 0, 0] = 0.0
        self.size = []
        for j in range(self.dim):
            self.size.append(size)
        self.size = tuple(self.size)

    def sample(self):
        coeff = torch.randn(*self.size, dtype=torch.cfloat)
        coeff = self.sqrt_eig * coeff
        u = torch.fft.ifftn(coeff)
        u = u.real
        return u


class NavierStokesDataset(AbstractDataset):
    def __init__(self, param, device='cpu', normalize=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params_eq = param
        self.sampler = GaussianRF(2, self.size, alpha=2.5, tau=7)
        self.dt = 1e-3
        self.device = device
        self.coords = get_mgrid(self.size, vmin=0, vmax=0.5, dim=2)
        self.coord_dim = self.coords.shape[-1]

        # Normalize the dataset optionally. Calculate values first.
        # self.min_value, self.max_value, self.mean_value, self.std_value = self.calc_min_max_mean_std_values()
        self.min_value, self.max_value, self.mean_value, self.std_value = 0, 0, 0, 0
        self.normalize = False

    def navier_stokes_2d(self, w0, f, visc, T, delta_t, record_steps):
        # Grid size - must be power of 2
        N = w0.size()[-1]
        # Maximum frequency
        k_max = math.floor(N / 2.0)
        # Number of steps to final time
        steps = math.ceil(T / delta_t)
        # Initial vorticity to Fourier space
        w_h = torch.fft.fftn(w0, (N, N))
        # Forcing to Fourier space
        f_h = torch.fft.fftn(f, (N, N))
        # If same forcing for the whole batch
        if len(f_h.size()) < len(w_h.size()):
            f_h = torch.unsqueeze(f_h, 0)
        # Record solution every this number of steps
        record_time = math.floor(steps / record_steps)
        # Wavenumbers in y-direction
        k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device),
                         torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N, 1)
        # Wavenumbers in x-direction
        k_x = k_y.transpose(0, 1)
        # Negative Laplacian in Fourier space
        lap = 4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2)
        lap[0, 0] = 1.0
        # Dealiasing mask
        dealias = torch.unsqueeze(
            torch.logical_and(torch.abs(k_y) <= (2.0 / 3.0) * k_max, torch.abs(k_x) <= (2.0 / 3.0) * k_max).float(), 0)
        # Saving solution and time
        sol = torch.zeros(*w0.size(), record_steps, 1, device=w0.device, dtype=torch.float)
        sol_t = torch.zeros(record_steps, device=w0.device)
        # Record counter
        c = 0
        # Physical time
        t = 0.0
        for j in range(steps):
            if j % record_time == 0:
                # Solution in physical space
                w = torch.fft.ifftn(w_h, (N, N))
                # Record solution and time
                sol[..., c, 0] = w.real
                # sol[...,c,1] = w.imag
                sol_t[c] = t
                c += 1
            # Stream function in Fourier space: solve Poisson equation
            psi_h = w_h.clone()
            psi_h = psi_h / lap
            # Velocity field in x-direction = psi_y
            q = psi_h.clone()
            temp = q.real.clone()
            q.real = -2 * math.pi * k_y * q.imag
            q.imag = 2 * math.pi * k_y * temp
            q = torch.fft.ifftn(q, (N, N))
            # Velocity field in y-direction = -psi_x
            v = psi_h.clone()
            temp = v.real.clone()
            v.real = 2 * math.pi * k_x * v.imag
            v.imag = -2 * math.pi * k_x * temp
            v = torch.fft.ifftn(v, (N, N))
            # Partial x of vorticity
            w_x = w_h.clone()
            temp = w_x.real.clone()
            w_x.real = -2 * math.pi * k_x * w_x.imag
            w_x.imag = 2 * math.pi * k_x * temp
            w_x = torch.fft.ifftn(w_x, (N, N))
            # Partial y of vorticity
            w_y = w_h.clone()
            temp = w_y.real.clone()
            w_y.real = -2 * math.pi * k_y * w_y.imag
            w_y.imag = 2 * math.pi * k_y * temp
            w_y = torch.fft.ifftn(w_y, (N, N))
            # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
            F_h = torch.fft.fftn(q * w_x + v * w_y, (N, N))
            # Dealias
            F_h = dealias * F_h
            # Cranck-Nicholson update
            w_h = (-delta_t * F_h + delta_t * f_h + (1.0 - 0.5 * delta_t * visc * lap) * w_h) / \
                  (1.0 + 0.5 * delta_t * visc * lap)
            # Update real time (used only for recording)
            t += delta_t

        return sol, sol_t

    def _get_init_cond(self, index, start, end):
        print(f'generating {start}-{end-1} ICs')
        if self.buffer.get(f'init_cond_{index}') is None:
            w0s = []
            for i in range(start, end):
                torch.manual_seed(i if self.group != 'test' else self.max - i)
                w0 = self.sampler.sample().to(self.device)
                w0s.append(w0)
            w0 = torch.stack(w0s, 0)

            state, _ = self.navier_stokes_2d(w0, f=self.params_eq['f'].to(self.device)
            , visc=self.params_eq['visc'], T=30,
                                             delta_t=self.dt, record_steps=20)
            init_cond = state[:, :, :, -1, 0].cpu()
            for i, ii in enumerate(range(start, end)):
                self.buffer[f'init_cond_{ii}'] = init_cond[i].numpy()
        else:
            init_cond = torch.from_numpy(torch.stack(self.buffer[f'init_cond_{i}'] for i in range(start, end)))

        return init_cond

    def _generate_trajectory(self, traj_id):
        batch_size_gen = 128
        start = traj_id // batch_size_gen * batch_size_gen
        end = start + batch_size_gen 
        if end > self.n_seq // self.n_seq_per_traj:
            end = self.n_seq // self.n_seq_per_traj
        print(f'generating {start}-{end-1}')
        with torch.no_grad():
            w0 = self._get_init_cond(traj_id, start, end).to(self.device)
            state, _ = self.navier_stokes_2d(w0, f=self.params_eq['f'].to(self.device)
            , visc=self.params_eq['visc'],
                                             T=self.t_horizon * self.n_seq_per_traj, delta_t=self.dt, record_steps=self.n * self.n_seq_per_traj)
        state = state.permute(0, 4, 3, 1, 2)
        for i, ii in enumerate(range(start, end)):
            self.buffer_shelve[f'{ii}'] = {'data': state[i].cpu().numpy()}

#################
#   SW sphere   #
#################


class ShallowWaterDataset(AbstractDataset):
    def __init__(self, root, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_path = os.path.join(root, f"shallow_water_{'test' if self.group == 'test' else 'train'}")
        self.files_obj_buf = dict()
        self._load_trajectory(0, file_object_only=True)
        coords_list = []
        if self.group == 'test_hr':
            phi = torch.tensor(self.files_obj_buf[0]['tasks/vorticity'].dims[1][0][:].ravel())
            theta = torch.tensor(self.files_obj_buf[0]['tasks/vorticity'].dims[2][0][:].ravel())
        else:
            phi = torch.tensor(self.files_obj_buf[0]['tasks/vorticity'].dims[1][0][:].ravel()[::2])
            theta = torch.tensor(self.files_obj_buf[0]['tasks/vorticity'].dims[2][0][:].ravel()[::2])

        spherical = get_mgrid_from_tensors([phi, theta])
        phi_vert = spherical[..., 0]
        theta_vert = spherical[..., 1]
        r = 1
        x = torch.cos(phi_vert) * torch.sin(theta_vert) * r
        y = torch.sin(phi_vert) * torch.sin(theta_vert) * r
        z = torch.cos(theta_vert) * r
        coords_list.append(torch.stack([x, y, z], dim=-1))

        self.coords_ang = get_mgrid_from_tensors([phi, theta])
        self.coords = torch.cat(coords_list, dim=-1).float()
        self.coord_dim = self.coords.shape[-1]

    def _load_trajectory(self, traj_id, file_object_only=False):
        if self.files_obj_buf.get(traj_id) is None:
            self.files_obj_buf[traj_id] = h5py.File(os.path.join(self.dataset_path, f'traj_{traj_id:04d}.h5'), mode='r')
        if file_object_only:
            return
        f = self.files_obj_buf[traj_id]
        if self.group == 'test_hr':
            return {'data': np.stack([
                f['tasks/height'][...] * 3000.,
                f['tasks/vorticity'][...] * 2,
                ], axis=0)}
        return {'data': np.stack([
            f['tasks/height'][:, ::2, ::2] * 3000.,
            f['tasks/vorticity'][:, ::2, ::2] * 2,
            ], axis=0)}
    
def extract_data(fp, variables):
    loaded_file = netCDFDataset(fp, 'r')
    data_dict = {}
    for var in variables:
        data_dict[var] = loaded_file.variables[var][:].data
    return data_dict


#################
# Cahn-Hilliard #
#################

class CahnHilliardDataset(AbstractDataset):
    def __init__(self, param, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eqs = CahnHilliardPDE()
        self.coords = get_mgrid(self.size, vmin=-1.0, vmax=1.0, dim=2)
        self.grid = CartesianGrid([[-1., 1.]] * 2, self.size, periodic=True)

        assert self.n_seq_per_traj == 1, 'n_seq_per_traj must be 1 for Cahn-Hilliard.'

        self.min_value, self.max_value, self.mean_value, self.std_value = None, None, None, None
        self.normalize = False

    def _get_init_cond(self, index):
        np.random.seed(index if self.group != 'test' else self.max - index)
        u = ScalarField.random_uniform(UnitGrid([self.size, self.size]), -1, 1)
        return u

    def _generate_trajectory(self, traj_id):
        print(f'generating {traj_id}')
        storage = MemoryStorage()
        state = self._get_init_cond(traj_id)
        self.eqs.solve(state, t_range=self.t_horizon * self.n_seq_per_traj, dt=1e-2, tracker=storage.tracker(self.dt_eval))
        # Stack on time dimension
        data = np.stack(storage.data, axis=0)

        # Clear storage
        storage.clear()

        # Unsqueeze channel dimension
        data = data[None, :, :, :]

        # Discard the first 5 timesteps, they are too noisy
        data = data[:, 10:, :, :]

        self.buffer_shelve[f'{traj_id}'] = {'data': data}


##############
# Diffusion  #
##############


class DiffusionDataset(AbstractDataset):
    def __init__(self, param, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eqs = DiffusionPDE(0.1)
        self.grid = CartesianGrid([[-3, 3], [-3, 3]], [64, 64], periodic=False)
        self.coords = get_mgrid(self.size, vmin=-3.0, vmax=3.0, dim=2)

        assert self.n_seq_per_traj == 1, 'n_seq_per_traj must be 1 for Diffusion PDE.'

        # self.min_value, self.max_value, self.mean_value, self.std_value = self.calc_min_max_mean_std_values()
        self.normalize = False

    def _get_init_cond(self, index):
        np.random.seed(index if self.group != 'test' else self.max - index)

        state = ScalarField(self.grid)

        # Get a random point x between -2 and 2, and y between 0 and 2.
        if self.group != 'test':
            x = np.random.rand() * 4 - 2
            y = np.random.rand() * 2

        # Get a random point x between -2 and 2, and y between -2 and 0.
        else:
            x = np.random.rand() * 4 - 2
            y = - np.random.rand() * 2

        value = np.random.rand() * 0.5 + 5.0
        state.insert([x, y], value)
        return state

    def _generate_trajectory(self, traj_id):
        print(f'generating {traj_id}')
        storage = MemoryStorage()
        state = self._get_init_cond(traj_id)
        self.eqs.solve(state, t_range=16, dt=0.01, tracker=storage.tracker(self.dt_eval))
        # Stack on time dimension
        data = np.stack(storage.data, axis=0)

        # Unsqueeze channel dimension
        data = data[None, :, :, :]

        # Discard the first timestep, its too noisy. Last one is also discarded.
        data = data[:, 7:, :, :]
        data = data[:, :20, :, :]

        self.buffer_shelve[f'{traj_id}'] = {'data': data}


#######################
# Diffusion on sphere #
#######################


class DiffusionSphereDataset(AbstractDataset):
    def __init__(self, param, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eqs = DiffusionPDE(0.1)

        assert self.n_seq_per_traj == 1, 'n_seq_per_traj must be 1 for Diffusion PDE.'

        # Parameters
        Nphi = 128
        Ntheta = 64
        dealias = 3 / 2
        R = 1
        self.stop_sim_time = 100
        dtype = np.float64

        # Bases
        coords = d3.S2Coordinates("phi", "theta")
        self.dist = d3.Distributor(coords, dtype=dtype)
        self.basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)

        # phi [0-2pi] in 256, theta [0-pi] in 128
        self.phi, self.theta = self.dist.local_grids(self.basis)

        # Create meshgrid for the sphere, shape (128, 256, 2), [phi is idx 0, theta is idx 1]
        self.grid = np.stack(np.meshgrid(self.phi, self.theta, indexing='ij'), axis=-1)

        # Set as coordinate grid
        self.coords = np.array(self.grid.reshape(-1, 2))

        # self.min_value, self.max_value, self.mean_value, self.std_value = self.calc_min_max_mean_std_values()
        self.normalize = False

    def _gauss_peak_on_sphere(self, phi_0, theta_0, sigma=0.25):
        # Get phi, theta
        phi = self.grid[:, :, 0]
        theta = self.grid[:, :, 1]

        d = np.arccos((np.sin(theta) * np.cos(phi) * np.sin(theta_0) * np.cos(phi_0) + \
                       np.sin(theta) * np.sin(phi) * np.sin(theta_0) * np.sin(phi_0) + \
                       np.cos(theta) * np.cos(theta_0)) / np.sqrt(
            (((np.sin(theta) * np.cos(phi)) ** 2 + (np.sin(theta) * np.sin(phi)) ** 2 + np.cos(theta) ** 2) * \
             ((np.sin(theta_0) * np.cos(phi_0)) ** 2 + (np.sin(theta_0) * np.sin(phi_0)) ** 2 + np.cos(theta_0) ** 2))))

        h0 = np.exp(-d ** 2 / (2 * sigma ** 2))
        return h0

    def _get_init_cond(self, index):
        np.random.seed(index if self.group != 'test' else self.max - index)

        # Sample random point on the sphere
        theta_0 = np.random.rand() * 2 * np.pi
        phi_0 = np.arccos(1 - 2 * np.random.rand())

        # Create field
        h = self.dist.Field(name="h", bases=self.basis)

        # Initial conditions: gaussian peak at a random point on the sphere
        h["g"] = self._gauss_peak_on_sphere(phi_0, theta_0)

        return h

    def _generate_trajectory(self, traj_id):

        # Get initial condition
        h = self._get_init_cond(traj_id)
        D = 0.01
        problem = d3.IVP([h], namespace=locals())
        problem.add_equation("dt(h) - D*lap(h) = 0")

        # Solver
        solver = problem.build_solver(d3.RK222)
        solver.stop_sim_time = self.stop_sim_time

        # Set up storage
        h_list = [np.copy(h["g"])]
        t_list = [solver.sim_time]

        while solver.proceed:
            solver.step(self.dt_eval)
            if (solver.iteration - 1) % 10 == 0:
                h.change_scales(1)
                h_list.append(np.copy(h["g"]))
                t_list.append(solver.sim_time)

        # Unsqueeze channel dimension
        data = np.array(h_list)[None, :, :, :]

        # Save first 20 steps
        data = data[:, :20, :, :]

        self.buffer_shelve[f'{traj_id}'] = {'data': data}


##########################
# SW on sphere rand init #
##########################


class ShallowWaterRandInit(AbstractDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.n_seq_per_traj == 1, 'n_seq_per_traj must be 1 for Diffusion PDE.'

        # Simulation units
        self.meter = meter = 1 / 6.37122e6
        self.hour = hour = 1
        self.second = second = hour / 3600
        self.stop_sim_time = stop_sim_time = 360 * hour

        # Parameters
        Nphi = 192
        Ntheta = 96
        dealias = 3/2
        R = 6.37122e6 * meter
        g = 9.80616 * meter / second**2
        Omega = 7.292e-5 / second
        dtype = np.float64

        self.coords_dedalus = coords = d3.S2Coordinates('phi', 'theta')
        self.dist = dist = d3.Distributor(coords, dtype=dtype)
        self.basis = basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)

        # phi [0-2pi] in 256, theta [0-pi] in 128
        self.phi, self.theta = self.dist.local_grids(self.basis)

        # Create meshgrid for the sphere, shape (128, 256, 2), [phi is idx 0, theta is idx 1]
        self.grid = np.stack(np.meshgrid(self.phi, self.theta, indexing='ij'), axis=-1)

        self.coords = np.array(self.grid.reshape(-1, 2))

        # Initialize a solver once to get the grid
        u = dist.VectorField(coords, name='u', bases=basis)
        h = dist.Field(name='h', bases=basis)

        # Substitutions
        zcross = lambda A: d3.MulCosine(d3.skew(A))

        # Initial conditions: zonal jet
        phi, theta = dist.local_grids(basis)
        lat = np.pi / 2 - theta + 0*phi
        umax = 80 * meter / second
        lat0 = np.pi / 7
        lat1 = np.pi / 2 - lat0
        en = np.exp(-4 / (lat1 - lat0)**2)
        jet = (lat0 <= lat) * (lat <= lat1)
        u_jet = umax / en * np.exp(1 / (lat[jet] - lat0) / (lat[jet] - lat1))
        u['g'][0][jet]  = u_jet

        # Initial conditions: balanced height
        c = dist.Field(name='c')
        problem = d3.LBVP([h, c], namespace=locals())
        problem.add_equation("g*lap(h) + c = - div(u@grad(u) + 2*Omega*zcross(u))")
        problem.add_equation("ave(h) = 0")
        solver = problem.build_solver()
        solver.solve()

        self._u = u
        self._h = h
        self._c = c

    def _get_init_cond(self, index):
        np.random.seed(index if self.group != 'test' else self.max - index)

        lat2 = np.pi / 4

        #categorical distribution over longitude
        hpert = 120 * self.meter + 30 * self.meter * (1 - 2 * np.random.rand())

        alpha = 1 / 3 + 1 / 9 * (1 - 2 * np.random.rand())
        beta = 1 / 15 + 1 / 45 * (1 - 2 * np.random.rand())

        h = self.dist.Field(name='h', bases=self.basis)
        phi, theta = self.dist.local_grids(self.basis)
        lat = np.pi / 2 - theta + 0*phi
        h['g'] = self._h['g'] + (hpert * np.cos(lat) * np.exp(-(phi/alpha)**2) * np.exp(-((lat2-lat)/beta)**2))

        return self._u, h

    def _generate_trajectory(self, traj_id):
        
        Omega = 7.292e-5 / self.second
        nu = 1e5 * self.meter**2 / self.second / 32**2 # Hyperdiffusion matched at ell=32
        g = 9.80616 * self.meter / self.second**2
        H = 1e4 * self.meter
        zcross = lambda A: d3.MulCosine(d3.skew(A))
        timestep = 1200 * self.second

        u, h = self._get_init_cond(traj_id)

        # Get initial condition
        problem = d3.IVP([u, h], namespace=locals())
        problem.add_equation("dt(u) + nu*lap(lap(u)) + g*grad(h) + 2*Omega*zcross(u) = - u@grad(u)")
        problem.add_equation("dt(h) + nu*lap(lap(h)) + H*div(u) = - div(h*u)")

        # Solver
        solver = problem.build_solver(d3.RK222)
        solver.stop_sim_time = self.stop_sim_time

        # Set up storage
        h_list = [np.copy(h["g"])]
        u_list = [np.copy(u["g"])]
        t_list = [solver.sim_time]

        while solver.proceed:
            solver.step(timestep)
            if (solver.iteration - 1) % 50 == 0:
                h.change_scales(1)
                u.change_scales(1)
                h_list.append(np.copy(h["g"]))
                u_list.append(np.copy(u["g"]))
                t_list.append(solver.sim_time)
                logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))

        # Unsqueeze channel dimension
        data = np.concatenate([np.array(h_list[1:21])[:, None, :, :], np.array(u_list[1:21])[:, :, :]], axis=1).transpose(1, 0, 2, 3)

        self.buffer_shelve[f'{traj_id}'] = {'data': data}

    def __getitem__(self, index):
        data, coords, index = super().__getitem__(index)

        # Skip first 6 frames, they are too noisy
        data = data[6:, :, :, :]

        return data, coords, index


class ShallowWaterRandInitHalfRes(ShallowWaterRandInit):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Simulation units
        self.meter = meter = 1 / 6.37122e6
        self.hour = hour = 1
        self.second = second = hour / 3600
        self.stop_sim_time = stop_sim_time = 360 * hour

        Nphi = 192 // 2
        Ntheta = 96 // 2
        dealias = 3/2
        R = 6.37122e6 * meter
        dtype = np.float64

        self.coords_dedalus = coords = d3.S2Coordinates('phi', 'theta')
        self.dist = dist = d3.Distributor(coords, dtype=dtype)
        self.basis = basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)

        # phi [0-2pi] in 256, theta [0-pi] in 128
        self.phi, self.theta = self.dist.local_grids(self.basis)

        # Create meshgrid for the sphere, shape (128, 256, 2), [phi is idx 0, theta is idx 1]
        self.grid = np.stack(np.meshgrid(self.phi, self.theta, indexing='ij'), axis=-1)

        self.coords = np.array(self.grid.reshape(-1, 2))

    def __getitem__(self, item):
        data, coords, index = super().__getitem__(item)

        # Create strided window [T,H,W,C] -> [T,H-1,W-1,2,2,C]
        strided_window = np.lib.stride_tricks.sliding_window_view(
            data,
            window_shape=(2, 2),
            axis=(1, 2)
        )

        # Avg over the window [T,H-1,W-1,2,2,C] -> [T,H-1,W-1,C]
        avg_pooled_data = np.mean(strided_window[:, ::2, ::2], axis=(4, 5))

        return avg_pooled_data, coords, index


###########################################
# Internally heated convection (ball IVP) #
###########################################


class InternallyHeatedConvection(AbstractDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.n_seq_per_traj == 1, 'n_seq_per_traj must be 1 for Diffusion PDE.'
        dtype = np.float64
        mesh = None
        # Bases
        coords = d3.SphericalCoordinates('phi', 'theta', 'r')
        dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
        ball = d3.BallBasis(coords, shape=self.size, radius=1, dealias=3/2, dtype=dtype)

        # Store the coordinates.
        phi, theta, r = dist.local_grids(ball)
        self.phi = phi
        self.theta = theta
        self.r = r
        # Meshgrid
        self.coords = np.meshgrid(phi, theta, r, indexing='ij')
        self.grid = np.stack(np.meshgrid(phi, theta, r, indexing='ij'), axis=-1)

    def _generate_trajectory(self, traj_id):
        # Generate disjoint train and test sets
        seed = traj_id if self.group != 'test' else self.max - traj_id

        # Parameters
        Nphi, Ntheta, Nr = self.size
        Rayleigh = 1e6
        Prandtl = 1
        dealias = (1, 1, 1)
        stop_sim_time = 12
        timestepper = d3.SBDF2
        max_timestep = 0.02
        min_timestep = 1e-4
        dtype = np.float64
        mesh = None

        # Bases
        coords = d3.SphericalCoordinates('phi', 'theta', 'r')
        dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
        ball = d3.BallBasis(coords, shape=(Nphi, Ntheta, Nr), radius=1, dealias=dealias, dtype=dtype)
        sphere = ball.surface

        # Fields
        u = dist.VectorField(coords, name='u',bases=ball)
        p = dist.Field(name='p', bases=ball)
        T = dist.Field(name='T', bases=ball)
        tau_p = dist.Field(name='tau_p')
        tau_u = dist.VectorField(coords, name='tau u', bases=sphere)
        tau_T = dist.Field(name='tau T', bases=sphere)

        # Substitutions
        phi, theta, r = dist.local_grids(ball)
        r_vec = dist.VectorField(coords, bases=ball.radial_basis)
        r_vec['g'][2] = r
        T_source = 6
        kappa = (Rayleigh * Prandtl)**(-1/2)
        nu = (Rayleigh / Prandtl)**(-1/2)
        lift = lambda A: d3.Lift(A, ball, -1)
        strain_rate = d3.grad(u) + d3.trans(d3.grad(u))
        shear_stress = d3.angular(d3.radial(strain_rate(r=1), index=1))

        # Problem
        problem = d3.IVP([p, u, T, tau_p, tau_u, tau_T], namespace=locals())
        problem.add_equation("div(u) + tau_p = 0")
        problem.add_equation("dt(u) - nu*lap(u) + grad(p) - r_vec*T + lift(tau_u) = - cross(curl(u),u)")
        problem.add_equation("dt(T) - kappa*lap(T) + lift(tau_T) = - u@grad(T) + kappa*T_source")
        problem.add_equation("shear_stress = 0")  # Stress free
        problem.add_equation("radial(u(r=1)) = 0")  # No penetration
        problem.add_equation("radial(grad(T)(r=1)) = -2")
        problem.add_equation("integ(p) = 0")  # Pressure gauge

        # Solver
        solver = problem.build_solver(timestepper)
        solver.stop_sim_time = stop_sim_time

        # Initial conditions
        T.fill_random('g', seed=traj_id, distribution='normal', scale=0.1) # Random noise
        T.low_pass_filter(scales=0.5)
        T['g'] += 1 - r**2 # Add equilibrium state
        file_handler_mode = 'overwrite'
        initial_timestep = max_timestep

        # CFL
        CFL = d3.CFL(solver, initial_timestep, cadence=10, safety=0.5, threshold=0.1, max_dt=max_timestep, min_dt=min_timestep)
        CFL.add_velocity(u)

        T_list = [np.copy(T['g'])]

        logger.info('Starting main loop')
        while solver.proceed:
            timestep = CFL.compute_timestep()
            solver.step(timestep)
            if (solver.iteration-1) % 10 == 0:
                T.change_scales(1)
                T_list.append(np.copy(T['g']))
                logger.info("Iteration=%i, Time=%e, dt=%e" %(solver.iteration, solver.sim_time, timestep))


        # Unsqueeze channel dimension
        data = np.array(T_list[10:37])[:, None, :, :].transpose(1, 0, 2, 3, 4)
        self.buffer_shelve[f'{traj_id}'] = {'data': data}

    def __getitem__(self, index):
        data, coords, index = super().__getitem__(index)

        # Skip first 6 frames, they are too noisy
        data = data[6:, :, :, :]

        return data, coords, index
