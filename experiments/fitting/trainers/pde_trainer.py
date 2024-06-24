import jax
import optax
import jax.numpy as jnp
from functools import partial

from experiments.fitting.trainers.trainer_utils.solvers import _solve_latent_ode
from experiments.fitting.trainers._base_pde_trainer import BasePDETrainer

from enf.latents.autodecoder_meta import PositionOrientationFeatureAutodecoderMeta


class MetaSGDPDETrainer(BasePDETrainer):

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

        if config.dataset.name in ["diff_sphere", "shallow_water", "shallow_water_low_res"]:
            coordinate_system = "polar"
        elif config.dataset.name == "ihc":
            coordinate_system = "ball"
        else:
            coordinate_system = "cartesian"

        # Used in case the NeF is optimised with states from a single step
        self.inner_autodecoder_single_step = PositionOrientationFeatureAutodecoderMeta(
            num_signals=config.dataset.batch_size,
            num_latents=config.nef.num_latents,
            latent_dim=config.nef.latent_dim,
            num_pos_dims=nef.cross_attn_invariant.num_z_pos_dims,
            num_ori_dims=nef.cross_attn_invariant.num_z_ori_dims,
            gaussian_window_size=config.nef.gaussian_window,
            coordinate_system=coordinate_system
        )
        # Used in case the NeF is being optimised with states from the entire trajectory
        self.inner_autodecoder_full_traj = PositionOrientationFeatureAutodecoderMeta(
            num_signals=config.dataset.batch_size * config.training.nef.fit_on_num_steps,
            num_latents=config.nef.num_latents,
            latent_dim=config.nef.latent_dim,
            num_pos_dims=nef.cross_attn_invariant.num_z_pos_dims,
            num_ori_dims=nef.cross_attn_invariant.num_z_ori_dims,
            gaussian_window_size=config.nef.gaussian_window,
            coordinate_system=coordinate_system
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
        self.meta_sgd_opt = optax.adam(self.config.meta.learning_rate_meta_sgd)
        self.ode_opt = optax.chain(optax.clip_by_global_norm(1.0),
                                   optax.adamw(self.config.optimizer.learning_rate_enf))

        # Random key
        key = jax.random.PRNGKey(self.seed)

        # Split key
        key, nef_key = jax.random.split(key)
        key, autodecoder_key = jax.random.split(key)
        key, inner_autodecoder_key = jax.random.split(key)
        key, ode_key = jax.random.split(key)

        # Create a test batch to get the shape of the latent space
        autodecoder_params = self.outer_autodecoder.init(autodecoder_key)
        p, a, window = self.outer_autodecoder.apply(autodecoder_params)

        # Initialize learning rates for the autodecoder
        lr_pos = jnp.ones((1)) * self.config.meta.inner_learning_rate_p
        lr_a = jnp.ones((a.shape[-1])) * self.config.meta.inner_learning_rate_a
        lr_gaussian_window = jnp.ones((1)) * self.config.meta.inner_learning_rate_window

        # Put lrs in frozendict
        meta_sgd_lrs = {
            'p_pos': lr_pos,
            'a': lr_a,
            'gaussian_window': lr_gaussian_window
        }

        # Add orientation learning rate if we have orientation dimensions
        if self.outer_autodecoder.num_ori_dims > 0:
            lr_ori = jnp.ones((1)) * self.config.meta.inner_learning_rate_p
            meta_sgd_lrs['p_ori'] = lr_ori

        # Initialize nef
        sample_coords = jax.random.normal(nef_key, (1, 128, self.config.nef.num_in))
        nef_params = self.nef.init(nef_key, sample_coords[:, :self.config.training.max_num_sampled_points], p, a,
                                   window)

        # Init ode model
        ode_params = self.ode_model.init(ode_key, (p, a, window))

        # Init but discard the params, we only care about the optimizer state
        _ = self.inner_autodecoder.init(inner_autodecoder_key)
        _ = self.inner_autodecoder_single_step.init(inner_autodecoder_key)

        # Create train state
        train_state = self.TrainState(
            params={'nef': nef_params, 'autodecoder': autodecoder_params, 'meta_sgd_lrs': meta_sgd_lrs, 'ode_params': ode_params},
            nef_opt_state=self.nef_opt.init(nef_params),
            autodecoder_opt_state=self.autodecoder_opt.init(autodecoder_params),
            meta_sgd_opt_state=self.meta_sgd_opt.init(meta_sgd_lrs),
            ode_opt_state=self.ode_opt.init(ode_params),
            rng=key
        )
        return train_state

    @partial(jax.jit, static_argnums=(0, 4, 5))
    def inner_loop(self, outer_params, outer_state, initial_state, autodecoder, initial_state_dp=0.):
        """Performs the inner loop of the meta learning algorithm.

        Args:
            outer_params: The parameters of the outer loop.
            outer_state: The state of the outer loop.
            initial_state: The initial state.
            initial_state_dp: The initial state dropout probability, used for evaluating sensitivity to sparsity of
                initial state.
        """
        img = jnp.reshape(initial_state, (initial_state.shape[0], -1, initial_state.shape[-1]))

        # Select coords
        coords = self.coords

        # In case we're subsamplling the initial state, we first subsample the coordinates and image.
        if initial_state_dp > 0:
            dp_mask = jax.random.permutation(
                outer_state.rng,
                self.coords.shape[0]
            )[:int(coords.shape[0] * initial_state_dp)]
            coords = coords[dp_mask]
            img = img[:, dp_mask]

        # Generate random mask of coordinates, one for every inner step
        mask = jax.random.permutation(
            outer_state.rng,
            jnp.broadcast_to(jnp.arange(coords.shape[0])[:, jnp.newaxis],
                             (coords.shape[0], self.config.meta.num_inner_steps + 1)),
            independent=True,
        )
        mask = mask[:self.config.training.max_num_sampled_points, :]

        # Broadcast autodecoder params over the batch dimension
        inner_autodecoder_params = jax.tree_map(
            lambda p: jnp.repeat(p, img.shape[0], axis=0), outer_params['autodecoder']
        )

        # Randomly sample positions
        if self.config.meta.noise_pos_inner_loop:
            inner_autodecoder_params['params']['p_pos'] = inner_autodecoder_params['params']['p_pos'] + (
                        jax.random.normal(
                            outer_state.rng,
                            inner_autodecoder_params['params']['p_pos'].shape,
                        ) * self.config.meta.noise_pos_inner_loop)

        # Create inner state
        inner_state = outer_state.replace(
            params={'nef': outer_params['nef'], 'autodecoder': inner_autodecoder_params,
                    'meta_sgd_lrs': outer_params['meta_sgd_lrs']}
        )

        def loss_fn(params, masked_coords, masked_img):
            """Loss function for the inner loop.

            Args:
                params: The current parameters of the model.
                masked_coords: The masked coordinates.
                masked_img: The masked image.
            """
            p, a, window = autodecoder.apply(params['autodecoder'])
            out = self.nef.apply(params['nef'], masked_coords, p, a, window)
            return jnp.mean((out - masked_img) ** 2)

        # Create inner grad fn
        inner_grad_fn = jax.grad(loss_fn)

        # Do inner loop
        for inner_step in range(self.config.meta.num_inner_steps):
            # Mask the coordinates and labels
            masked_coords = coords[mask[:, inner_step]]
            masked_img = img[:, mask[:, inner_step]]

            # Broadcast the coordinates over the batch dimension
            masked_coords = jnp.broadcast_to(masked_coords, (img.shape[0], *masked_coords.shape))

            # Get inner grads and update inner autodecoder params
            inner_grad = inner_grad_fn(
                inner_state.params,
                masked_coords=masked_coords,
                masked_img=masked_img,
            )

            # Scale the gradient by the batch size, we need to do this since we're taking the mean of the loss.
            inner_grad['autodecoder'] = jax.tree_map(lambda x: x * img.shape[0], inner_grad['autodecoder'])

            # Optionally zero out gaussian window param updates.
            if not self.config.nef.optimize_gaussian_window and 'gaussian_window' in inner_grad['autodecoder']['params']:
                inner_grad['autodecoder']['params']['gaussian_window'] = jnp.zeros_like(
                    inner_grad['autodecoder']['params']['gaussian_window'])

            # Scale inner grads by the learning rates
            inner_autodecoder_updates = jax.tree_util.tree_map_with_path(
                lambda path, grad: - inner_state.params['meta_sgd_lrs'][path[-1].key] * grad,
                inner_grad['autodecoder'])

            updated_inner_params = optax.apply_updates(inner_state.params['autodecoder'], inner_autodecoder_updates)
            inner_state = inner_state.replace(
                params={'nef': inner_state.params['nef'], 'autodecoder': updated_inner_params,
                        'meta_sgd_lrs': inner_state.params['meta_sgd_lrs']})

        # Mask coordinates and labels for last step.
        masked_coords = coords[mask[:, inner_step + 1]]
        masked_img = img[:, mask[:, inner_step + 1]]

        # Broadcast the coordinates over the batch dimension
        masked_coords = jnp.broadcast_to(masked_coords, (img.shape[0], *masked_coords.shape))

        return loss_fn(
                inner_state.params,
                masked_coords=masked_coords,
                masked_img=masked_img,
            ), inner_state

    def _nef_train_step(self, state, batch):
        """ Performs a single enf train step, only updating gradients of enf based on reconstruction.

        Args:
            state (TrainState): The current training state.
            batch (dict): The current batch of data.

        Returns:
            TrainState: The updated training state.
        """
        # Unpack batch
        trajectory, _, _ = batch

        # Split random key
        inner_key, new_outer_key = jax.random.split(state.rng)
        outer_state = state.replace(rng=new_outer_key)

        # Calc enf train loss
        recon_loss, grads = jax.value_and_grad(self.enf_loss)(outer_state.params, outer_state, trajectory)

        # Update enf backbone
        nef_updates, nef_opt_state = self.nef_opt.update(grads['nef'], state.nef_opt_state, state.params['nef'])
        nef_params = optax.apply_updates(state.params['nef'], nef_updates)

        if self.config.optimizer.learning_rate_codes != 0:
            # Update autodecoder
            autodecoder_updates, autodecoder_opt_state = self.autodecoder_opt.update(grads['autodecoder'],
                                                                                     state.autodecoder_opt_state)
            autodecoder_params = optax.apply_updates(state.params['autodecoder'], autodecoder_updates)
        else:
            autodecoder_params = state.params['autodecoder']
            autodecoder_opt_state = state.autodecoder_opt_state

        meta_sgd_lr_updates, meta_sgd_opt_state = self.meta_sgd_opt.update(grads['meta_sgd_lrs'],
                                                                           state.meta_sgd_opt_state)
        meta_sgd_lrs = optax.apply_updates(state.params['meta_sgd_lrs'], meta_sgd_lr_updates)

        # Clip meta_sgd_lrs between 1e-6 and 10
        meta_sgd_lrs = jax.tree_map(lambda x: jnp.clip(x, 1e-6, 10), meta_sgd_lrs)

        # Return updated state, only updating the nef backbone
        return recon_loss, state.replace(
            params={'nef': nef_params,
                    'autodecoder': autodecoder_params,
                    'ode_params': state.params['ode_params'],
                    'meta_sgd_lrs': meta_sgd_lrs},
            nef_opt_state=nef_opt_state,
            autodecoder_opt_state=autodecoder_opt_state,
            meta_sgd_opt_state=meta_sgd_opt_state,
            ode_opt_state=state.ode_opt_state,
            rng=new_outer_key
        )

    def _ode_train_step(self, state, batch):
        # Unpack batch
        trajectory, _, _ = batch

        # Split random key
        inner_key, new_outer_key = jax.random.split(state.rng)
        outer_state = state.replace(rng=new_outer_key)

        # Get gradients for the outer loop and update params, gives grads for nef and ode
        recon_loss, grads = jax.value_and_grad(self.neural_ode_loss)(outer_state.params, outer_state, trajectory)

        # Update ODE model
        ode_updates, ode_opt_state = self.ode_opt.update(grads['ode_params'], state.ode_opt_state,
                                                         state.params['ode_params'])
        ode_params = optax.apply_updates(state.params['ode_params'], ode_updates)

        # Return updated state, only updating the ODE model
        return recon_loss, state.replace(
            params={'nef': state.params['nef'],
                    'autodecoder': state.params['autodecoder'],
                    'ode_params': ode_params,
                    'meta_sgd_lrs': state.params['meta_sgd_lrs']},
            nef_opt_state=state.nef_opt_state,
            autodecoder_opt_state=state.autodecoder_opt_state,
            meta_sgd_opt_state=state.meta_sgd_opt_state,
            ode_opt_state=ode_opt_state,
            rng=new_outer_key
        )

    def _dual_train_step(self, state, batch):
        # Unpack batch
        trajectory, _, _ = batch

        # Split random key
        inner_key, new_outer_key = jax.random.split(state.rng)
        outer_state = state.replace(rng=new_outer_key)

        # Get gradients for the outer loop and update params, gives grads for nef and ode
        recon_loss, grads = jax.value_and_grad(self.neural_ode_loss)(outer_state.params, outer_state,
                                                                     trajectory)

        # Update enf backbone
        nef_updates, nef_opt_state = self.nef_opt.update(grads['nef'], state.nef_opt_state, state.params['nef'])
        nef_params = optax.apply_updates(state.params['nef'], nef_updates)

        meta_sgd_lr_updates, meta_sgd_opt_state = self.meta_sgd_opt.update(grads['meta_sgd_lrs'],
                                                                           state.meta_sgd_opt_state)
        meta_sgd_lrs = optax.apply_updates(state.params['meta_sgd_lrs'], meta_sgd_lr_updates)

        # Clip meta_sgd_lrs between 1e-6 and 10
        meta_sgd_lrs = jax.tree_map(lambda x: jnp.clip(x, 1e-6, 10), meta_sgd_lrs)

        # Update ODE model
        ode_updates, ode_opt_state = self.ode_opt.update(grads['ode_params'], state.ode_opt_state,
                                                         state.params['ode_params'])
        ode_params = optax.apply_updates(state.params['ode_params'], ode_updates)

        # Return updated state, only updating the ODE model
        return recon_loss, state.replace(
            params={'nef': nef_params,
                    'autodecoder': state.params['autodecoder'],
                    'ode_params': ode_params,
                    'meta_sgd_lrs': meta_sgd_lrs},
            nef_opt_state=nef_opt_state,
            autodecoder_opt_state=state.autodecoder_opt_state,
            meta_sgd_opt_state=meta_sgd_opt_state,
            ode_opt_state=ode_opt_state,
            rng=new_outer_key
        )

    def _val_step(self, state, batch, initial_state_dp=0.):
        # Unpack batch
        trajectory, _, _ = batch

        # Limit to train+gen horizon
        trajectory = trajectory[:, :self.config.dataset.traj_len_train + self.config.dataset.traj_len_out_horizon]

        # Perform inner loop to get initial latents
        _, last_inner_state = self.inner_loop(
            outer_params=state.params,
            outer_state=state,
            initial_state=trajectory[:, 0],
            autodecoder=self.inner_autodecoder_single_step,
            initial_state_dp=initial_state_dp)

        # Obtain initial latents
        p_t0, a_t0, window_t0 = self.inner_autodecoder_single_step.apply(
            last_inner_state.params['autodecoder'])

        # Solve ode
        sol = _solve_latent_ode(
            f=lambda z, t: self.ode_model.apply(state.params['ode_params'], z),
            latents=(p_t0, a_t0, window_t0),
            t0=0,
            tf=self.config.dataset.traj_len_train + self.config.dataset.traj_len_out_horizon - 1,
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
        return (jnp.mean((recon[:, :self.config.dataset.traj_len_train] - trajectory[:, :self.config.dataset.traj_len_train]) ** 2),
                jnp.mean((recon[:, self.config.dataset.traj_len_train:] - trajectory[:, self.config.dataset.traj_len_train:]) ** 2))

    def create_functions(self):
        def ode_loss(params, state, trajectory):
            """Solves the ODE for the given initial latents and trajectory.

            Args:
                state (TrainState): The current training state.
                batch (dict): The current batch of data.
            """
            # Take the first image in the batch
            initial_state = trajectory[:, 0]

            # Limit trajectory length to 10, these are the train steps
            trajectory = trajectory[:, :self.config.dataset.traj_len_train]

            # Perform inner loop to get initial latents
            _, last_inner_state = self.inner_loop(params, state, initial_state, self.inner_autodecoder_single_step)

            # Obtain initial latents
            p_traj_in, a_traj_in, window_traj_in = self.inner_autodecoder_single_step.apply(last_inner_state.params['autodecoder'])

            # Unroll latents for 10 timesteps
            sol = _solve_latent_ode(
                f=lambda z, t: self.ode_model.apply(params['ode_params'], z),
                latents=(p_traj_in,
                         a_traj_in,
                         window_traj_in),
                t0=0,
                tf=self.config.dataset.traj_len_train - 1,
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

        def nef_loss(params, state, trajectory):
            # Perform inner loop to get initial latents
            if self.config.training.nef.fit_on_num_steps == 1:
                gt_state = trajectory[:, 0]
                inner_loss, last_inner_state = self.inner_loop(params, state, gt_state, self.inner_autodecoder_single_step)
                # Obtain initial latents
            else:
                # Subsample the trajectory
                key, new_key = jax.random.split(state.rng)
                idx = jax.random.permutation(new_key, jnp.arange(self.config.dataset.traj_len_train))[:self.config.training.nef.fit_on_num_steps]
                trajectory = trajectory[:, idx]

                gt_state = trajectory.reshape(trajectory.shape[0] * trajectory.shape[1], *trajectory.shape[2:])

                inner_loss, last_inner_state = self.inner_loop(params, state, gt_state, self.inner_autodecoder_full_traj)

            # Obtain initial latents
            # p_traj_in, a_traj_in, window_traj_in = self.inner_autodecoder_full_traj.apply(
            #     last_inner_state.params['autodecoder'])
            #
            # # Broadcast coordinates over batch dim
            # coords = jnp.broadcast_to(self.coords[None], (gt_state.shape[0], *self.coords.shape))
            #
            # # Flatten initial state
            # gt_state = jnp.reshape(gt_state, (gt_state.shape[0], -1, gt_state.shape[-1]))
            #
            # # Subsample the states
            # if self.config.training.max_num_sampled_points != -1:
            #     # Create a mask for coordinates [num_points]
            #     mask = jax.random.permutation(
            #         state.rng,
            #         # Arange over the number of points, broadcast over traj length
            #         jnp.broadcast_to(jnp.arange(coords.shape[1])[None, ...],
            #                          (coords.shape[0], coords.shape[1])),
            #         axis=1,
            #         independent=True,
            #     )
            #     mask = mask[:, :self.config.training.max_num_sampled_points]
            #
            #     gt_state = jax.vmap(lambda x, mask: x[mask], in_axes=(0, 0))(gt_state, mask)
            #     coords = jax.vmap(lambda x, mask: x[mask], in_axes=(0, 0))(coords, mask)
            #
            # # Forward pass through NeF
            # out = self.nef.apply(params['nef'], coords, p_traj_in, a_traj_in, window_traj_in)

            # Compute mse loss
            # return jnp.mean((out - gt_state) ** 2)
            return inner_loss

        # Train objectives
        self.neural_ode_loss = jax.jit(ode_loss)
        self.enf_loss = jax.jit(nef_loss)

        # For visualization
        self.apply_nef_jitted = jax.jit(self.nef.apply)

        # Different train steps, the correct one is selected in self.train_epoch
        self.dual_train_step = jax.jit(self._dual_train_step)
        self.nef_train_step = jax.jit(self._nef_train_step)
        self.ode_train_step = jax.jit(self._ode_train_step)

        # Validation step, only fit the latents
        # self.val_step = jax.jit(self._val_step)
        # self.val_step_dp5 = jax.jit(lambda state, batch: self._val_step(state, batch, initial_state_dp=0.05))
        # self.val_step_dp10 = jax.jit(lambda state, batch: self._val_step(state, batch, initial_state_dp=0.1))
        # self.val_step_dp50 = jax.jit(lambda state, batch: self._val_step(state, batch, initial_state_dp=0.5))

        self.val_step = self._val_step
        self.val_step_dp5 = lambda state, batch: self._val_step(state, batch, initial_state_dp=0.05)
        self.val_step_dp10 = lambda state, batch: self._val_step(state, batch, initial_state_dp=0.1)
        self.val_step_dp50 = lambda state, batch: self._val_step(state, batch, initial_state_dp=0.5)

    def visualize_batch(self, state, batch, name):
        """ Visualize the results of the model on a batch of data.

        Args:
            state: The current training state.
            batch: The current batch of data.
            name: The name of the plot.
            train: Whether we are training or validating.
        """

        # 1. Visualize the solutions of the ODE model
        trajectory, _, _ = batch
        # Take the first image in the batch
        initial_state = trajectory[:, 0]

        # Limit to 20 steps
        trajectory = trajectory[:, :self.config.dataset.traj_len_train + self.config.dataset.traj_len_out_horizon]

        # Perform inner loop to get initial latents
        _, last_inner_state = self.inner_loop(state.params, state, initial_state, self.inner_autodecoder_single_step)

        # Obtain initial latents
        p_traj_in, a_traj_in, window_traj_in = self.inner_autodecoder_single_step.apply(
            last_inner_state.params['autodecoder'])

        # Unroll latents for 20 timesteps
        p_traj_hat, a_traj_hat, window_traj_hat = _solve_latent_ode(
            f=lambda z, t: self.ode_model.apply(state.params['ode_params'], z),
            latents=(p_traj_in,
                     a_traj_in,
                     window_traj_in),
            t0=0,
            tf=self.config.dataset.traj_len_train + self.config.dataset.traj_len_out_horizon - 1,
            h=self.config.node.dt,
            method=self.config.node.method
        )

        # Log reconstructions
        self.visualize_and_log(trajectory, state, p_traj_hat=p_traj_hat, a_traj_hat=a_traj_hat,
                               window_traj_hat=window_traj_hat, name=name + "pred/")

        # 2. Perform sanity check of equivariance properties
        if self.epoch > self.config.test.test_equiv_at_epoch and not self.equivariance_sanity_checked:
            # Log transformed reconstructions (equivariance sanity check)
            self.transform_visualize_and_log_2d(trajectory, state, p_traj_hat=p_traj_hat, a_traj_hat=a_traj_hat,
                                   window_traj_hat=window_traj_hat, name=name + "pred/")
            self.equivariance_sanity_checked = True

        # # 3. Perform subsampling visualization
        # _, last_inner_state = self.inner_loop(
        #     state.params,
        #     state,
        #     initial_state,
        #     self.inner_autodecoder_single_step,
        #     initial_state_dp=0.5
        # )
        #
        # # Obtain initial latents
        # p_traj_in, a_traj_in, window_traj_in = self.inner_autodecoder_single_step.apply(
        #     last_inner_state.params['autodecoder'])
        #
        # # Unroll latents for 20 timesteps
        # p_traj_hat, a_traj_hat, window_traj_hat = _solve_latent_ode(
        #     f=lambda z, t: self.ode_model.apply(state.params['ode_params'], z),
        #     latents=(p_traj_in,
        #              a_traj_in,
        #              window_traj_in),
        #     t0=0,
        #     tf=self.config.dataset.traj_len_train + self.config.dataset.traj_len_out_horizon - 1,
        #     h=self.config.node.dt,
        #     method=self.config.node.method
        # )
        #
        # # Log reconstructions
        # self.visualize_and_log(trajectory, state, p_traj_hat=p_traj_hat, a_traj_hat=a_traj_hat,
        #                        window_traj_hat=window_traj_hat, name='subsampling-0.5/' + name + "pred/")
        #
        # # 3. Perform subsampling visualization
        # _, last_inner_state = self.inner_loop(
        #     state.params,
        #     state,
        #     initial_state,
        #     self.inner_autodecoder_single_step,
        #     initial_state_dp=0.1
        # )
        #
        # # Obtain initial latents
        # p_traj_in, a_traj_in, window_traj_in = self.inner_autodecoder_single_step.apply(
        #     last_inner_state.params['autodecoder'])
        #
        # # Unroll latents for 20 timesteps
        # p_traj_hat, a_traj_hat, window_traj_hat = _solve_latent_ode(
        #     f=lambda z, t: self.ode_model.apply(state.params['ode_params'], z),
        #     latents=(p_traj_in,
        #              a_traj_in,
        #              window_traj_in),
        #     t0=0,
        #     tf=self.config.dataset.traj_len_train + self.config.dataset.traj_len_out_horizon - 1,
        #     h=self.config.node.dt,
        #     method=self.config.node.method
        # )
        #
        # # Log reconstructions
        # self.visualize_and_log(trajectory, state, p_traj_hat=p_traj_hat, a_traj_hat=a_traj_hat,
        #                        window_traj_hat=window_traj_hat, name='subsampling-0.1/' + name + "pred/")

        # # 4 perform visualization of 2 random states in the inner loop
        # # Subsample the trajectory
        # key, new_key = jax.random.split(state.rng)
        # idx = jax.random.permutation(new_key, jnp.arange(self.config.dataset.traj_len_train))[
        #       :self.config.training.nef.fit_on_num_steps]
        # trajectory_su = trajectory[:, idx]
        #
        # gt_state = trajectory_su.reshape(trajectory_su.shape[0] * trajectory_su.shape[1], *trajectory_su.shape[2:])
        #
        # inner_loss, last_inner_state = self.inner_loop(state.params, state, gt_state, self.inner_autodecoder_full_traj)
        # # Obtain latents
        # p_traj, a_traj, window_traj = self.inner_autodecoder_full_traj.apply(
        #     last_inner_state.params['autodecoder'])
        #
        # # Unflatten time and batch dimension
        # p_traj = p_traj.reshape(*trajectory_su.shape[:2], *p_traj.shape[1:])
        # a_traj = a_traj.reshape(*trajectory_su.shape[:2], *a_traj.shape[1:])
        # window_traj = window_traj.reshape(*trajectory_su.shape[:2], *window_traj.shape[1:])
        #
        # # Log reconstructions
        # self.visualize_and_log(trajectory_su, state, p_traj_hat=p_traj, a_traj_hat=a_traj,
        #                        window_traj_hat=window_traj, name='inner-loop/' + name + "pred/")
