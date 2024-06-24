import jax
import optax
import jax.numpy as jnp
from functools import partial
import wandb
from flax import struct, core
from typing import Any

from experiments.fitting.trainers.trainer_utils.solvers import _solve_latent_ode
from experiments.fitting.trainers._base_pde_trainer import BasePDETrainer

from enf.latents.autodecoder import PositionOrientationFeatureAutodecoder


class NonMetaPDETrainer(BasePDETrainer):
    class TrainState(struct.PyTreeNode):
        params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

        nef_opt_state: optax.OptState = struct.field(pytree_node=True)
        autodecoder_opt_state: optax.OptState = struct.field(pytree_node=True)
        ode_opt_state: optax.OptState = struct.field(pytree_node=True)

        rng: jnp.ndarray = struct.field(pytree_node=True)

    def __init__(self, config, nef, ode_model, train_loader, val_loader, coords, seed=42):
        super().__init__(
            config=config,
            nef=nef,
            ode_model=ode_model,
            train_loader=train_loader,
            val_loader=val_loader,
            coords=coords,
            seed=seed
        )

        # The autodecoder is
        self.autodecoder_single_step = PositionOrientationFeatureAutodecoder(
            num_signals=config.dataset.num_signals_train,
            num_latents=config.nef.num_latents,
            latent_dim=config.nef.latent_dim,
            num_pos_dims=nef.cross_attn_invariant.num_z_pos_dims,
            num_ori_dims=nef.cross_attn_invariant.num_z_ori_dims,
            gaussian_window_size=config.nef.gaussian_window,
            coordinate_system="polar" if config.dataset.name in ["diff_sphere", "shallow_water"] else "cartesian"
        )
        self.val_autodecoder_single_step = PositionOrientationFeatureAutodecoder(
            num_signals=config.dataset.num_signals_test,
            num_latents=config.nef.num_latents,
            latent_dim=config.nef.latent_dim,
            num_pos_dims=nef.cross_attn_invariant.num_z_pos_dims,
            num_ori_dims=nef.cross_attn_invariant.num_z_ori_dims,
            gaussian_window_size=config.nef.gaussian_window,
            coordinate_system="polar" if config.dataset.name in ["diff_sphere", "shallow_water"] else "cartesian"
        )

    def init_train_state(self):
        """ Initialize the training state.

        Returns:
            TrainState: The initialized training state.
        """

        self.nef_opt = optax.chain(
        optax.clip_by_global_norm(1.0),
           optax.adamw(self.config.optimizer.learning_rate_enf),
        )
        self.autodecoder_opt = optax.adam(self.config.optimizer.learning_rate_codes)
        self.ode_opt = optax.chain(optax.clip_by_global_norm(1.0),
                                   optax.adamw(self.config.optimizer.learning_rate_enf))

        # Random key
        key = jax.random.PRNGKey(self.seed)

        # Split key
        key, nef_key = jax.random.split(key)
        key, autodecoder_key = jax.random.split(key)
        key, ode_key = jax.random.split(key)

        # Create a test batch to get the shape of the latent space
        autodecoder_params = self.autodecoder_single_step.init(autodecoder_key, jnp.array([0]))
        p, a, window = self.autodecoder_single_step.apply(autodecoder_params, jnp.array([0]))

        # Initialize nef
        sample_coords = jax.random.normal(nef_key, (1, 128, self.config.nef.num_in))
        nef_params = self.nef.init(nef_key, sample_coords[:, :self.config.training.max_num_sampled_points], p, a,
                                   window)

        # Init ode model
        ode_params = self.ode_model.init(ode_key, (p, a, window))

        # Create train state
        train_state = self.TrainState(
            params={'nef': nef_params, 'autodecoder': autodecoder_params, 'ode_params': ode_params},
            nef_opt_state=self.nef_opt.init(nef_params),
            autodecoder_opt_state=self.autodecoder_opt.init(autodecoder_params),
            ode_opt_state=self.ode_opt.init(ode_params),
            rng=key
        )
        return train_state

    def _nef_train_step(self, state, batch, mask, autodecoder_fn):
        """ Performs a single enf train step, only updating gradients of enf based on reconstruction.

        Args:
            state (TrainState): The current training state.
            batch (dict): The current batch of data.

        Returns:
            TrainState: The updated training state.
        """
        # Unpack batch
        trajectory, _, traj_idx = batch

        # Split random key
        inner_key, new_outer_key = jax.random.split(state.rng)

        # Calc enf train loss
        recon_loss, grads = jax.value_and_grad(self.enf_loss)(state.params, state, autodecoder_fn, trajectory, mask, traj_idx)

        # Update enf backbone
        nef_updates, nef_opt_state = self.nef_opt.update(grads['nef'], state.nef_opt_state, state.params['nef'])
        nef_params = optax.apply_updates(state.params['nef'], nef_updates)

        # Autodecoder updates
        autodecoder_updates, autodecoder_opt_state = self.autodecoder_opt.update(grads['autodecoder'], state.autodecoder_opt_state, state.params['autodecoder'])
        autodecoder_params = optax.apply_updates(state.params['autodecoder'], autodecoder_updates)

        # Return updated state, only updating the nef backbone
        return recon_loss, state.replace(
            params={'nef': nef_params,
                    'autodecoder': autodecoder_params,
                    'ode_params': state.params['ode_params']},
            nef_opt_state=nef_opt_state,
            autodecoder_opt_state=autodecoder_opt_state,
            ode_opt_state=state.ode_opt_state,
            rng=new_outer_key
        )

    def _nef_train_step_autodec_only(self, state, batch, mask, autodecoder_fn):
        """ Performs a single enf train step, only updating gradients of enf based on reconstruction.

        Args:
            state (TrainState): The current training state.
            batch (dict): The current batch of data.

        Returns:
            TrainState: The updated training state.
        """
        # Unpack batch
        trajectory, _, traj_idx = batch

        # Split random key
        inner_key, new_outer_key = jax.random.split(state.rng)

        # Calc enf train loss
        recon_loss, grads = jax.value_and_grad(self.enf_loss)(state.params, state, autodecoder_fn, trajectory, mask, traj_idx)

        # Autodecoder updates
        autodecoder_updates, autodecoder_opt_state = self.autodecoder_opt.update(grads['autodecoder'], state.autodecoder_opt_state, state.params['autodecoder'])
        autodecoder_params = optax.apply_updates(state.params['autodecoder'], autodecoder_updates)

        # Return updated state, only updating the nef backbone
        return recon_loss, state.replace(
            params={'nef': state.params['nef'],
                    'autodecoder': autodecoder_params,
                    'ode_params': state.params['ode_params']},
            nef_opt_state=state.nef_opt_state,
            autodecoder_opt_state=autodecoder_opt_state,
            ode_opt_state=state.ode_opt_state,
            rng=new_outer_key
        )

    def _ode_train_step(self, state, batch):
        # Unpack batch
        trajectory, _, traj_idx = batch

        # Split random key
        inner_key, new_outer_key = jax.random.split(state.rng)
        outer_state = state.replace(rng=new_outer_key)

        # Get gradients for the outer loop and update params, gives grads for nef and ode
        recon_loss, grads = jax.value_and_grad(self.neural_ode_loss)(outer_state.params, outer_state, trajectory, traj_idx)

        # Update ODE model
        ode_updates, ode_opt_state = self.ode_opt.update(grads['ode_params'], state.ode_opt_state,
                                                         state.params['ode_params'])
        ode_params = optax.apply_updates(state.params['ode_params'], ode_updates)

        # Return updated state, only updating the ODE model
        return recon_loss, state.replace(
            params={'nef': state.params['nef'],
                    'autodecoder': state.params['autodecoder'],
                    'ode_params': ode_params,},
            nef_opt_state=state.nef_opt_state,
            autodecoder_opt_state=state.autodecoder_opt_state,
            ode_opt_state=ode_opt_state,
            rng=new_outer_key
        )

    @partial(jax.jit, static_argnums=(0, 3,))
    def _val_step(self, state, batch, autodecoder_fn):
        # Unpack batch
        trajectory, _, traj_idx = batch

        # Limit to 20 steps
        trajectory = trajectory[:, :20]

        # Obtain initial latents
        p_t0, a_t0, window_t0 = autodecoder_fn(
            state.params['autodecoder'], traj_idx)

        # Solve ode
        sol = _solve_latent_ode(
            f=lambda z, t: self.ode_model.apply(state.params['ode_params'], z),
            latents=(p_t0, a_t0, window_t0),
            t0=0,
            tf=19,
            h=self.config.node.dt,
            method=self.config.node.method
        )

        # Loop over the trajectory and get the reconstructions
        (p_traj_hat_fl, a_traj_hat_fl, window_traj_hat_fl) = jax.tree_map(lambda x: jnp.reshape(x, (-1, *x.shape[2:])), sol)

        # Broadcast coords over batch dimension
        coords = jnp.broadcast_to(self.coords, (p_traj_hat_fl.shape[0], *self.coords.shape))

        recon = []
        # Loop over all coords in chunks of self.config.training.max_num_sampled_pixels
        for i in range(0, coords.shape[1], self.config.training.max_num_sampled_points):
            out = self.apply_nef_jitted(state.params['nef'],
                                        coords[:, i:i + self.config.training.max_num_sampled_points], p_traj_hat_fl,
                                        a_traj_hat_fl, window_traj_hat_fl)
            recon.append(out)
        recon = jnp.concatenate(recon, axis=1)

        # Unflatten time and batch dimension
        recon = recon.reshape(*trajectory.shape)

        # Calculate mse in and out t.
        return jnp.mean((recon[:, :10] - trajectory[:, :10]) ** 2), jnp.mean((recon[:, 10:] - trajectory[:, 10:]) ** 2)

    def create_functions(self):
        def ode_loss(params, state, trajectory, traj_idx):
            """Solves the ODE for the given initial latents and trajectory.

            Args:
                state (TrainState): The current training state.
                batch (dict): The current batch of data.
            """
            # Limit trajectory length to 10, these are the train steps
            trajectory = trajectory[:, :10]

            # Obtain initial latents
            p_traj_in, a_traj_in, window_traj_in = self.autodecoder_single_step.apply(state.params['autodecoder'], traj_idx)

            # Unroll latents for 10 timesteps
            sol = _solve_latent_ode(
                f=lambda z, t: self.ode_model.apply(params['ode_params'], z),
                latents=(p_traj_in,
                         a_traj_in,
                         window_traj_in),
                t0=0,
                tf=9,
                h=self.config.node.dt,
                method=self.config.node.method
            )

            # Flatten the latents
            (p_traj_hat_fl, a_traj_hat_fl, window_traj_hat_fl) = jax.tree_map(
                lambda x: jnp.reshape(x, (-1, *x.shape[2:])), sol)

            if self.config.training.max_num_sampled_points < self.coords.shape[0]:
                # Create a mask for coordinates [num_points]
                mask = jax.random.permutation(
                    state.rng,
                    # Arange over the number of points, broadcast over traj length
                    jnp.broadcast_to(jnp.arange(self.coords.shape[0])[None, ...],
                                     (trajectory.shape[1], self.coords.shape[0])),
                    axis=1,
                    independent=True,
                )
                mask = mask[:, :self.config.training.max_num_sampled_points]  # Only take the first max_num_sampled_points

                # Broadcast over traj len and gather the coordinates
                coords = jax.vmap(lambda x, mask: x[mask], in_axes=(None, 0))(self.coords, mask)

                # Broadcast over batch dimension, flatten spatial dims
                coords = jnp.broadcast_to(coords[None], (trajectory.shape[0], *coords.shape))
                coords = jnp.reshape(coords, (trajectory.shape[0] * trajectory.shape[1], -1, coords.shape[-1]))

                # Flatten trajectory, mask and gather the values
                trajectory_fl = jnp.reshape(trajectory,
                                            (trajectory.shape[0], trajectory.shape[1], -1, trajectory.shape[-1]))
                trajectory_m = jax.vmap(jax.vmap(lambda x, mask: x[mask.squeeze()], in_axes=(0, 0)), in_axes=(0, None))(trajectory_fl,
                                                                                                         mask[..., None])
                trajectory_m = jnp.reshape(trajectory_m, (trajectory.shape[0] * trajectory.shape[1], -1, trajectory.shape[-1]))
            else:
                coords = jnp.broadcast_to(self.coords[None, None], (trajectory.shape[0], trajectory.shape[1], *self.coords.shape))
                coords = jnp.reshape(coords, (trajectory.shape[0] * trajectory.shape[1], -1, coords.shape[-1]))
                trajectory_m = jnp.reshape(trajectory, (trajectory.shape[0] * trajectory.shape[1], -1, trajectory.shape[-1]))

            # Get the output of the NeF
            recon = self.nef.apply(params['nef'], coords, p_traj_hat_fl, a_traj_hat_fl, window_traj_hat_fl)

            # Compute mse loss
            return jnp.mean((recon - trajectory_m) ** 2)

        def nef_loss(params, state, autodecoder_fn, trajectory, mask, traj_idx):
            # Select the first state in the trajectory
            initial_state = trajectory[:, 0]

            # Flatten initial state
            initial_state = jnp.reshape(initial_state, (initial_state.shape[0], -1, initial_state.shape[-1]))

            # Broadcast coordinates over batch dim
            coords = jnp.broadcast_to(self.coords[None], (initial_state.shape[0], *self.coords.shape))

            # Mask coordinates
            if mask is not None:
                initial_state = initial_state[:, mask]
                coords = coords[:, mask]

            # Create another mask for max_num_sampled_points
            if self.config.training.max_num_sampled_points < coords.shape[1]:
                mask = jax.random.permutation(
                    state.rng,
                    # Arange over the number of points, broadcast over traj length
                    coords.shape[1]
                )
                mask = mask[:self.config.training.max_num_sampled_points]
                initial_state = initial_state[:, mask]
                coords = coords[:, mask]

            # Obtain initial latents
            p_traj_in, a_traj_in, window_traj_in = autodecoder_fn(params['autodecoder'], traj_idx)

            # Forward pass through NeF
            out = self.nef.apply(params['nef'], coords, p_traj_in, a_traj_in, window_traj_in)

            # Compute mse loss
            return jnp.mean((out - initial_state) ** 2)

        # Train objectives
        self.neural_ode_loss = jax.jit(ode_loss)
        self.enf_loss = jax.jit(nef_loss, static_argnums=(2))

        # For visualization
        self.apply_nef_jitted = jax.jit(self.nef.apply)

        # Different train steps, the correct one is selected in self.train_epoch
        self.nef_train_step = jax.jit(self._nef_train_step, static_argnums=(3,))
        self.nef_train_step_autodec_only = jax.jit(self._nef_train_step_autodec_only, static_argnums=(3,))
        # self.nef_train_step_autodec_only = self._nef_train_step_autodec_only
        self.ode_train_step = jax.jit(self._ode_train_step)

        self.val_step = self._val_step

    def train_epoch(self, state):
        """ Train the model for one epoch.

        Args:
            state: The current training state.
            epoch: The current epoch.
        """
        # Check which train step we are using
        if self.epoch > self.config.training.nef.train_from_epoch and self.epoch <= self.config.training.nef.train_until_epoch:
            self.train_nef = True
        else:
            self.train_nef = False

        if self.epoch > self.config.training.ode.train_from_epoch and self.epoch <= self.config.training.ode.train_until_epoch:
            self.train_ode = True
        else:
            self.train_ode = False

        # Loop over batches
        loss_ep = 0
        for batch_idx, batch in enumerate(self.train_loader):
            if self.train_nef:
                loss, state = self.nef_train_step(state, batch, None, self.autodecoder_single_step.apply)
            elif self.train_ode:
                loss, state = self.ode_train_step(state, batch)
            loss_ep += loss

            # Log every n steps
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                wandb.log({'mse_step': loss})
                self.update_prog_bar(step=batch_idx)

            # Increment global step
            self.global_step += 1

        # Update epoch loss
        self.metrics['train_mse_epoch'] = loss_ep / len(self.train_loader)
        wandb.log({'train_mse_epoch': self.metrics['train_mse_epoch']}, commit=False)
        return state

    def validate_epoch(self, state):
        """ Validates the model. Since we're doing autodecoding, requires
            training a validation autodecoder from scratch.

        Args:
            state: The current training state.
        Returns:
            state: The updated training state.
        """

        # temporarily store all coordinates
        all_coords = self.coords

        val_metrics = {}

        # Loop over the train set to obtain train mse in t
        train_mse_in_t = 0
        train_mse_out_t = 0
        self.global_val_step = 0

        for batch_idx, batch in enumerate(self.train_loader):
            mse_in_t, mse_out_t = self.val_step(state, batch, self.autodecoder_single_step.apply)
            train_mse_in_t += mse_in_t
            train_mse_out_t += mse_out_t

            # Log every n steps
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                self.update_prog_bar(step=batch_idx, train=True)

            # Increment global val step
            self.global_val_step += 1
        val_metrics['train_mse_in_t_sc'] = train_mse_in_t / len(self.train_loader)
        val_metrics['train_mse_out_t_sc'] = train_mse_out_t / len(self.train_loader)

        # Perform validation for different dp rates.
        for dp in [0.0, 0.05, 0.1, 0.5]:
            # Initialize autodecoder
            key, init_key = jax.random.split(state.rng)
            autodecoder_params = self.val_autodecoder_single_step.init(init_key, jnp.ones(3, dtype=jnp.int32))

            # Create validation state
            val_state = state.replace(
                params={'nef': state.params['nef'], 'autodecoder': autodecoder_params, 'ode_params': state.params['ode_params']},
                autodecoder_opt_state=self.autodecoder_opt.init(autodecoder_params),
                rng=key
            )

            # Set total val epochs
            self.total_val_epochs = self.config.training.nef.train_until_epoch
            self.global_val_step = 0

            # Create a mask
            if dp > 0:
                dp_mask = jax.random.permutation(
                    state.rng,
                    all_coords.shape[0],
                )[:int(all_coords.shape[0] * dp)]
            else:
                dp_mask = None

            # Fitting latents to the validation set -- no dropout
            for epoch in range(1, self.total_val_epochs):
                losses = 0
                self.val_epoch = epoch

                for batch_idx, batch in enumerate(self.val_loader):
                    loss, val_state = self.nef_train_step_autodec_only(val_state, batch, dp_mask, self.val_autodecoder_single_step.apply)
                    losses += loss

                    # Log every n steps
                    if batch_idx % self.config.logging.log_every_n_steps == 0:
                        self.metrics['val_loss'] = loss
                        self.update_prog_bar(step=batch_idx, train=False)

                    # Increment global val step
                    self.global_val_step += 1

            # Loop over the dataset again, this time to get unroll loss
            val_mse_in_t, val_mse_out_t = 0, 0
            for batch_idx, batch in enumerate(self.val_loader):
                # unpack batch
                traj, _, traj_idx = batch
                mse_in_t, mse_out_t = self.val_step(val_state, batch, self.val_autodecoder_single_step.apply)
                val_mse_in_t += mse_in_t
                val_mse_out_t += mse_out_t

                # Log every n steps
                if batch_idx % self.config.logging.log_every_n_steps == 0:
                    self.update_prog_bar(step=batch_idx, train=False)

                # Increment global val step
                self.global_val_step += 1
            if dp > 0:
                val_metrics[f'val_mse_in_t_dp{dp}'] = val_mse_in_t / len(self.val_loader)
                val_metrics[f'val_mse_out_t_dp{dp}'] = val_mse_out_t / len(self.val_loader)
            else:
                val_metrics['val_mse_in_t'] = val_mse_in_t / len(self.val_loader)
                val_metrics['val_mse_out_t'] = val_mse_out_t / len(self.val_loader)

            self.visualize_batch(val_state, self.val_autodecoder_single_step.apply, batch, f"val_dp{dp}/")

            # Do the same for train
            key, init_key = jax.random.split(state.rng)
            autodecoder_params = self.autodecoder_single_step.init(init_key, jnp.ones(3, dtype=jnp.int32))

            train_state = state.replace(
                params={'nef': state.params['nef'], 'autodecoder': autodecoder_params, 'ode_params': state.params['ode_params']},
                autodecoder_opt_state=self.autodecoder_opt.init(autodecoder_params),
                rng=key
            )
            for epoch in range(1, self.total_val_epochs):
                for batch_idx, batch in enumerate(self.train_loader):
                    _, train_state = self.nef_train_step_autodec_only(train_state, batch, dp_mask, self.autodecoder_single_step.apply)
                    # Log every n steps
                    if batch_idx % self.config.logging.log_every_n_steps == 0:
                        self.update_prog_bar(step=batch_idx, train=True)

            # Loop over the dataset again, this time to get unroll loss
            train_mse_in_t, train_mse_out_t = 0, 0
            for batch_idx, batch in enumerate(self.train_loader):
                # unpack batch
                traj, _, traj_idx = batch
                mse_in_t, mse_out_t = self.val_step(train_state, batch, self.autodecoder_single_step.apply)
                train_mse_in_t += mse_in_t
                train_mse_out_t += mse_out_t

                # Log every n steps
                if batch_idx % self.config.logging.log_every_n_steps == 0:
                    self.update_prog_bar(step=batch_idx, train=True)

                # Increment global val step
                self.global_val_step += 1
            if dp > 0:
                val_metrics[f'train_mse_in_t_dp{dp}'] = train_mse_in_t / len(self.train_loader)
                val_metrics[f'train_mse_out_t_dp{dp}'] = train_mse_out_t / len(self.train_loader)
            else:
                val_metrics['train_mse_in_t'] = train_mse_in_t / len(self.train_loader)
                val_metrics['train_mse_out_t'] = train_mse_out_t / len(self.train_loader)

            self.visualize_batch(train_state, self.autodecoder_single_step.apply, batch, f"tr_dp{dp}/")

        # Log metrics
        wandb.log(val_metrics)

        # Reset val epoch
        self.val_epoch = 0
        self.total_val_epochs = 0
        self.global_val_step = 0

        return val_state

    def visualize_batch(self, state, autodecoder_fn, batch, name):
        """ Visualize the results of the model on a batch of data.

        Args:
            state: The current training state.
            batch: The current batch of data.
            name: The name of the plot.
            train: Whether we are training or validating.
        """
        trajectory, _, traj_idx = batch

        # Limit to 20 steps
        trajectory = trajectory[:, :20]

        # Perform inner loop to get initial latents
        p_traj_in, a_traj_in, window_traj_in = autodecoder_fn(
            state.params['autodecoder'], traj_idx)

        # Unroll latents for 20 timesteps
        p_traj_hat, a_traj_hat, window_traj_hat = _solve_latent_ode(
            f=lambda z, t: self.ode_model.apply(state.params['ode_params'], z),
            latents=(p_traj_in,
                     a_traj_in,
                     window_traj_in),
            t0=0,
            tf=19,
            h=self.config.node.dt,
            method=self.config.node.method
        )

        # Log reconstructions
        self.visualize_and_log(trajectory, state, p_traj_hat=p_traj_hat, a_traj_hat=a_traj_hat,
                               window_traj_hat=window_traj_hat, name=name + "pred/")
