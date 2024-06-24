import wandb
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

import tqdm

# For trainstate
import jax
from flax import struct, core
import optax
import jax.numpy as jnp

# Checkpointing
import orbax.checkpoint as ocp
from omegaconf import OmegaConf

# Autodecoder
from enf.latents.autodecoder_meta import PositionOrientationFeatureAutodecoderMeta


class BasePDETrainer:

    class TrainState(struct.PyTreeNode):
        params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

        nef_opt_state: optax.OptState = struct.field(pytree_node=True)
        autodecoder_opt_state: optax.OptState = struct.field(pytree_node=True)
        ode_opt_state: optax.OptState = struct.field(pytree_node=True)

        meta_sgd_opt_state: optax.OptState = struct.field(pytree_node=True)

        rng: jnp.ndarray = struct.field(pytree_node=True)

    def __init__(
            self,
            config,
            nef,
            ode_model,
            coords,
            train_loader,
            val_loader,
            seed=42
    ):
        # Store models
        self.nef = nef
        self.ode_model = ode_model

        if config.dataset.name in ["diff_sphere", "shallow_water", "shallow_water_low_res"]:
            coordinate_system = "polar"
        elif config.dataset.name == "ihc":
            coordinate_system = "ball"
        else:
            coordinate_system = "cartesian"

        # Set inner and outer autodecoders
        self.outer_autodecoder = PositionOrientationFeatureAutodecoderMeta(
            num_signals=1,  # Since we're doing meta-learning, we only optimize one set of latents
            num_latents=config.nef.num_latents,
            latent_dim=config.nef.latent_dim,
            num_pos_dims=nef.cross_attn_invariant.num_z_pos_dims,
            num_ori_dims=nef.cross_attn_invariant.num_z_ori_dims,
            gaussian_window_size=config.nef.gaussian_window,
            coordinate_system=coordinate_system
        )
        self.inner_autodecoder = PositionOrientationFeatureAutodecoderMeta(
            num_signals=config.dataset.batch_size * config.dataset.traj_len_train,
            # Since we're doing meta-learning, the inner and val autodecoders have batch_size as num signals
            num_latents=config.nef.num_latents,
            latent_dim=config.nef.latent_dim,
            num_pos_dims=nef.cross_attn_invariant.num_z_pos_dims,
            num_ori_dims=nef.cross_attn_invariant.num_z_ori_dims,
            gaussian_window_size=config.nef.gaussian_window,
            coordinate_system=coordinate_system
        )

        # Store config
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.seed = seed
        self.coords = coords

        # Number of samples to log
        self.num_logged_samples = min(2, self.config.dataset.batch_size)
        self.equivariance_sanity_checked = False

        # Placeholders for train and val steps
        self.train_step = None
        self.val_step = None

        # Keep track of training state
        self.global_step = 0
        self.epoch = 0

        # Keep track of state of validation
        self.global_val_step = 0

        # Keep track of metrics
        self.metrics = {}

        # Set train step
        self.train_nef = False
        self.train_ode = False

        # Description strings for train and val progress bars
        self.prog_bar_desc = """{state} (optim {optim}) :: epoch - {epoch}/{total_epochs} | step - {step}/{global_step} ::"""
        self.prog_bar = tqdm.tqdm(
            desc=self.prog_bar_desc.format(
                state='Training',
                optim='-',
                epoch=self.epoch,
                total_epochs=self.config.training.num_epochs,
                step=0,
                global_step=self.global_step,
            ),
            total=len(self.train_loader)
        )

        # Set checkpoint options
        if self.config.logging.checkpoint:
            checkpoint_options = ocp.CheckpointManagerOptions(
                save_interval_steps=config.logging.checkpoint_every_n_epochs,
                max_to_keep=config.logging.keep_n_checkpoints,
            )
            self.checkpoint_manager = ocp.CheckpointManager(
                directory=config.logging.log_dir + '/checkpoints',
                options=checkpoint_options,
                item_handlers={
                    'state': ocp.StandardCheckpointHandler(),
                    'config': ocp.JsonCheckpointHandler(),
                },
                item_names=['state', 'config']
            )

    def update_prog_bar(self, step, train=True):
        """ Update the progress bar.

        Args:
            desc: The description string.
            loss: The current loss.
            epoch: The current epoch.
            step: The current step.
        """
        # If we are at the beginning of the epoch, reset the progress bar
        if step == 0:
            # Depending on whether we are training or validating, set the total number of steps
            if train:
                self.prog_bar.total = len(self.train_loader)
            else:
                self.prog_bar.total = len(self.val_loader)
            self.prog_bar.reset()
        else:
            self.prog_bar.update(self.config.logging.log_every_n_steps)

        epoch = self.epoch
        total_epochs = self.config.training.num_epochs

        if train:
            global_step = self.global_step
        else:
            global_step = self.global_val_step

        # Determine whether we are optimizing nef, ode or both
        if self.train_nef and self.train_ode:
            optim = 'nef+ode'
        elif self.train_nef:
            optim = 'nef'
        elif self.train_ode:
            optim = 'ode'
        else:
            optim = '-'

        # Update description string
        prog_bar_str = self.prog_bar_desc.format(
            state='Training' if train else 'Validation',
            epoch=epoch,
            optim=optim,
            total_epochs=total_epochs,
            step=step,
            global_step=global_step,
        )

        # Append metrics to description string
        if self.metrics:
            for key, value in self.metrics.items():
                prog_bar_str += f" -- {key} {value:.4f}"

        self.prog_bar.set_description_str(prog_bar_str)

    def save_checkpoint(self, state):
        """ Save the current state to a checkpoint

        Args:
            state: The current training state.
        """
        if self.config.logging.checkpoint:
            self.checkpoint_manager.save(step=self.epoch, args=ocp.args.Composite(
                state=ocp.args.StandardSave(state),
                config=ocp.args.JsonSave(OmegaConf.to_container(self.config))))

    def load_checkpoint(self):
        """ Load the latest checkpoint"""
        ckpt = self.checkpoint_manager.restore(self.checkpoint_manager.latest_step())

        # Load state
        loaded_state = self.TrainState(**ckpt.state)

        # Create surrogate train_state
        init_train_state = self.init_train_state()

        def create_opt_state(opt_state, loaded_opt_state):
            new_opt_state = []
            for idx, item in enumerate(opt_state):
                if isinstance(item, optax.EmptyState):
                    new_opt_state.append(item)
                elif item is None:
                    new_opt_state.append(None)
                elif isinstance(item, optax.ScaleByAdamState):
                    new_opt_state.append(optax.ScaleByAdamState(
                        count=loaded_opt_state[idx]['count'],
                        mu=loaded_opt_state[idx]['mu'],
                        nu=loaded_opt_state[idx]['nu'],
                    ))
                elif isinstance(item, tuple):
                    sub = create_opt_state(item, loaded_opt_state[idx])
                    new_opt_state.append(sub)
            return new_opt_state

        # Get opt states from surrogate train state
        for key in init_train_state.__dict__.keys():
            if key.endswith('opt_state'):
                opt_state = getattr(init_train_state, key)
                loaded_state = loaded_state.replace(**{key: create_opt_state(opt_state, getattr(loaded_state, key))})

        return loaded_state

    def train_model(self, num_epochs, state=None):
        """Trains the model for the given number of epochs.

        Args:
            num_epochs (int): The number of epochs to train for.

        Returns:
            state: The final training state.
        """

        # Keep track of global step
        self.global_step = 0
        self.epoch = 0

        if state is None:
            state = self.init_train_state()

        for epoch in range(1, num_epochs + 1):
            self.epoch = epoch
            wandb.log({'epoch': epoch}, commit=False)
            state = self.train_epoch(state)

            # Save checkpoint (ckpt manager takes care of saving every n epochs)
            self.save_checkpoint(state)

            # Validate every n epochs
            if epoch % self.config.test.test_interval == 0:
                self.validate_epoch(state)

            # Validate with dropout every m epochs
            if epoch % self.config.test.test_dp_interval == 0:
                self.validate_epoch_dp(state)
        return state

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

        # Set the train step
        if self.train_nef and self.train_ode:
            self.train_step = self.dual_train_step
        elif self.train_nef:
            self.train_step = self.nef_train_step
        elif self.train_ode:
            self.train_step = self.ode_train_step
        else:
            raise ValueError("No training step set")

        # Loop over batches
        loss_ep = 0
        for batch_idx, batch in enumerate(self.train_loader):
            loss, state = self.train_step(state, batch)
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
        """ Validates the model.

        Args:
            state: The current training state.
        """
        # Loop over validation set
        val_mse_in_t_ep, val_mse_out_t_ep = 0, 0
        for batch_idx, batch in enumerate(self.val_loader):
            val_mse_in_t, val_mse_out_t = self.val_step(state, batch)
            val_mse_in_t_ep += val_mse_in_t
            val_mse_out_t_ep += val_mse_out_t

            # Log every n steps
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                self.update_prog_bar(step=batch_idx, train=False)

        # Visualize last batch
        self.visualize_batch(state, batch, name='val/')

        # Update epoch loss
        self.metrics['val_mse_in_t'] = val_mse_in_t_ep / len(self.val_loader)
        self.metrics['val_mse_out_t'] = val_mse_out_t_ep / len(self.val_loader)
        wandb.log({'val_mse_in_t': self.metrics['val_mse_in_t'],
                   'val_mse_out_t': self.metrics['val_mse_out_t']}, commit=False)

        # Loop over train set
        train_mse_in_t_ep, train_mse_out_t_ep = 0, 0
        for batch_idx, batch in enumerate(self.train_loader):
            train_mse_in_t, train_mse_out_t = self.val_step(state, batch)
            train_mse_in_t_ep += train_mse_in_t
            train_mse_out_t_ep += train_mse_out_t

            # Log every n steps
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                self.update_prog_bar(step=batch_idx, train=False)

        # Visualize last batch
        self.visualize_batch(state, batch, name='tr/')

        # Update epoch loss
        self.metrics['train_mse_in_t'] = train_mse_in_t_ep / len(self.train_loader)
        self.metrics['train_mse_out_t'] = train_mse_out_t_ep / len(self.train_loader)
        wandb.log({'train_mse_in_t': self.metrics['train_mse_in_t'],
                   'train_mse_out_t': self.metrics['train_mse_out_t']}, commit=False)

    def validate_epoch_dp(self, state):
        # Loop over validation set
        val_mses = {
            "val_mse_in_t_dp5": 0,
            "val_mse_out_t_dp5": 0,
            "train_mse_in_t_dp5": 0,
            "train_mse_out_t_dp5": 0,
            "val_mse_in_t_dp10": 0,
            "val_mse_out_t_dp10": 0,
            "train_mse_in_t_dp10": 0,
            "train_mse_out_t_dp10": 0,
            "val_mse_in_t_dp50": 0,
            "val_mse_out_t_dp50": 0,
            "train_mse_in_t_dp50": 0,
            "train_mse_out_t_dp50": 0,
        }
        for batch_idx, batch in enumerate(self.val_loader):
            val_mse_in_t_dp5, val_mse_out_t_dp5 = self.val_step_dp5(state, batch)
            val_mse_in_t_dp10, val_mse_out_t_dp10 = self.val_step_dp10(state, batch)
            val_mse_in_t_dp50, val_mse_out_t_dp50 = self.val_step_dp50(state, batch)

            val_mses["val_mse_in_t_dp5"] += val_mse_in_t_dp5
            val_mses["val_mse_out_t_dp5"] += val_mse_out_t_dp5
            val_mses["val_mse_in_t_dp10"] += val_mse_in_t_dp10
            val_mses["val_mse_out_t_dp10"] += val_mse_out_t_dp10
            val_mses["val_mse_in_t_dp50"] += val_mse_in_t_dp50
            val_mses["val_mse_out_t_dp50"] += val_mse_out_t_dp50

            # Log every n steps
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                self.update_prog_bar(step=batch_idx, train=False)

        # Loop over train set
        for batch_idx, batch in enumerate(self.train_loader):

            train_mse_in_t_dp5, train_mse_out_t_dp5 = self.val_step_dp5(state, batch)
            train_mse_in_t_dp10, train_mse_out_t_dp10 = self.val_step_dp10(state, batch)
            train_mse_in_t_dp50, train_mse_out_t_dp50 = self.val_step_dp50(state, batch)

            val_mses["train_mse_in_t_dp5"] += train_mse_in_t_dp5
            val_mses["train_mse_out_t_dp5"] += train_mse_out_t_dp5
            val_mses["train_mse_in_t_dp10"] += train_mse_in_t_dp10
            val_mses["train_mse_out_t_dp10"] += train_mse_out_t_dp10
            val_mses["train_mse_in_t_dp50"] += train_mse_in_t_dp50
            val_mses["train_mse_out_t_dp50"] += train_mse_out_t_dp50

            # Log every n steps
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                self.update_prog_bar(step=batch_idx, train=False)

        # Mean over all batches
        for key in val_mses.keys():
            if "val" in key:
                val_mses[key] /= len(self.val_loader)
            else:
                val_mses[key] /= len(self.train_loader)

        # Log results to wandb.
        wandb.log(val_mses, commit=False)

    def visualize_and_log(self, *args, **kwargs):
        if self.config.dataset.name == "ihc":
            self._visualize_and_log_ball(*args, **kwargs)
        else:
            self._visualize_and_log_im(*args, **kwargs)

    def _visualize_and_log_ball(self, trajectory, state, p_traj_hat, a_traj_hat, window_traj_hat, name='recon'):
        """ Visualize and log the results.

        Args:
            state: The current training state.
        """

        p_traj_hat_flat = jnp.reshape(p_traj_hat, (-1, *p_traj_hat.shape[2:]))
        a_traj_hat_flat = jnp.reshape(a_traj_hat, (-1, *a_traj_hat.shape[2:]))
        window_traj_hat_flat = jnp.reshape(window_traj_hat, (-1, *window_traj_hat.shape[2:]))

        # Broadcast coords over batch dimension
        coords = jnp.broadcast_to(self.coords, (p_traj_hat_flat.shape[0], *self.coords.shape))

        out_all = None
        # Loop over all coords in chunks of self.config.training.max_num_sampled_pixels
        for i in range(0, coords.shape[1], self.config.training.max_num_sampled_points // 4):
            out = self.apply_nef_jitted(state.params['nef'],
                                        coords[:, i:i + self.config.training.max_num_sampled_points // 4],
                                        p_traj_hat_flat,
                                        a_traj_hat_flat, window_traj_hat_flat)

            if i == 0:
                out_all = out
            else:
                out_all = jnp.concatenate((out_all, out), axis=1)

        # Unflatten time and batch dimension
        out_all = out_all.reshape(*trajectory.shape)

        def spherical_to_euclidean(phi, theta, r):
            # Convert spherical coordinates to Cartesian coordinates for the plot
            x = np.sin(theta) * np.cos(phi) * r
            y = np.sin(theta) * np.sin(phi) * r
            z = np.cos(theta)
            return x, y, z

        # Meshgrid of polar coordinates
        phi_grid = jnp.linspace(0, 2 * jnp.pi, 48)
        theta_grid = jnp.linspace(0, jnp.pi, 24)
        r_grid = jnp.linspace(0, 1, 24)
        phi, theta, r = jnp.meshgrid(phi_grid, theta_grid, r_grid, indexing='ij')

        phi_slice = phi[:, phi.shape[1] // 2, :]
        theta_slice = theta[:, phi.shape[1] // 2, :]
        r_slice = r[:, phi.shape[1] // 2, :]
        x_slice, y_slice, z_slice = spherical_to_euclidean(phi_slice, theta_slice, r_slice)

        phi_sphere = phi[:, phi.shape[1] // 2:, -1]
        theta_sphere = theta[:, phi.shape[1] // 2:, -1]
        r_sphere = r[:, phi.shape[1] // 2:, -1]
        x_sphere, y_sphere, z_sphere = spherical_to_euclidean(phi_sphere, theta_sphere, r_sphere)

        # min-max normalize the trajectory
        out_all = (out_all - trajectory.min()) / (trajectory.max() - trajectory.min())
        trajectory = (trajectory - trajectory.min()) / (trajectory.max() - trajectory.min())

        # Select corresponding trajs [b, t, phi, theta, r, c]
        gt_sphere = trajectory[:, :, :, phi.shape[1] // 2:, -1, :]
        gt_slice = trajectory[:, :, :, phi.shape[1] // 2, :, :]
        pred_sphere = out_all[:, :, :, phi.shape[1] // 2:, -1, :]
        pred_slice = out_all[:, :, :, phi.shape[1] // 2, :, :]

        if "inner" not in name:
            # For the first image, plot slices
            fig, ax = plt.subplots(6, trajectory.shape[1] // 2, figsize=(25, 10))

            phi_gt_slice = trajectory[:, :, phi.shape[0] // 2, :, :, :]
            theta_gt_slice = trajectory[:, :, :, phi.shape[1] // 2, :, ]
            r_gt_slice = trajectory[:, :, :, :, phi.shape[1] // 2, ]

            phi_pred_slice = out_all[:, :, phi.shape[0] // 2, :, :, :]
            theta_pred_slice = out_all[:, :, :, phi.shape[1] // 2, :, :]
            r_pred_slice = out_all[:, :, :, :, phi.shape[1] // 2, :]

            for t in range(trajectory.shape[1] // 2):
                ax[0, t].imshow(phi_gt_slice[0, t].squeeze(), cmap='coolwarm')
                ax[0, t].axis('off')
                ax[0, t].set_title(f'T={t} / {np.mean((phi_gt_slice[0, t] - phi_pred_slice[0, t]) ** 2):.2E} ')
                ax[1, t].imshow(theta_gt_slice[0, t].squeeze(), cmap='coolwarm')
                ax[1, t].axis('off')
                ax[1, t].set_title(f'T={t} / {np.mean((theta_gt_slice[0, t] - theta_pred_slice[0, t]) ** 2):.2E} ')
                ax[2, t].imshow(r_gt_slice[0, t].squeeze(), cmap='coolwarm')
                ax[2, t].axis('off')
                ax[2, t].set_title(f'T={t} / {np.mean((r_gt_slice[0, t] - r_pred_slice[0, t]) ** 2):.2E} ')

                ax[3, t].imshow(phi_pred_slice[0, t].squeeze(), cmap='coolwarm')
                ax[3, t].axis('off')
                ax[4, t].imshow(theta_pred_slice[0, t].squeeze(), cmap='coolwarm')
                ax[4, t].axis('off')
                ax[5, t].imshow(r_pred_slice[0, t].squeeze(), cmap='coolwarm')
                ax[5, t].axis('off')

            wandb.log({f'slice-{name}': wandb.Image(fig)}, commit=True)

        for i, img in enumerate(out_all):
            if i >= self.num_logged_samples:
                break

            fig = plt.figure(figsize=(25, 10))
            for t in range(trajectory.shape[1] // 2):
                t = 2 * t
                ax0 = fig.add_subplot(2, trajectory.shape[1] // 2, 1 + t // 2, projection='3d')
                ax1 = fig.add_subplot(2, trajectory.shape[1] // 2, 1 + t // 2 + trajectory.shape[1] // 2, projection='3d')

                ax0.plot_surface(x_sphere, y_sphere, z_sphere, facecolors=plt.cm.magma(gt_sphere[i, t].squeeze()),
                                 rstride=1, cstride=1,
                                 shade=False, antialiased=False)
                ax0.plot_surface(x_slice, y_slice, z_slice, facecolors=plt.cm.magma(gt_slice[i, t].squeeze()), rstride=1, cstride=1, shade=False, antialiased=False)
                ax0.axis('off')
                ax0.set_title(f'T={t} / {np.mean((trajectory[i, t] - out_all[i, t]) ** 2):.2E} ')

                ax1.plot_surface(x_sphere, y_sphere, z_sphere, facecolors=plt.cm.magma(pred_sphere[i, t].squeeze()), rstride=1,
                                 cstride=1,
                                 shade=False, antialiased=False)
                ax1.plot_surface(x_slice, y_slice, z_slice, facecolors=plt.cm.magma(pred_slice[i, t].squeeze()), rstride=1, cstride=1,
                                 shade=False, antialiased=False)

                # Hide grid lines
                ax0.grid(False)
                ax1.grid(False)

                # Hide axes
                ax0.axis('off')
                ax1.axis('off')

                ax0.axes.set_xlim3d(left=-1, right=1)
                ax0.axes.set_ylim3d(bottom=-1, top=1)
                ax0.axes.set_zlim3d(bottom=-1, top=0)
                ax1.axes.set_xlim3d(left=-1, right=1)
                ax1.axes.set_ylim3d(bottom=-1, top=1)
                ax1.axes.set_zlim3d(bottom=-1, top=0)

                # Set equal scaling
                ax0.set_box_aspect([1, 1, 0.5])
                ax1.set_box_aspect([1, 1, 0.5])

            plt.tight_layout()
            # Disable axis
            wandb.log({f'sphere-{name}{i}': wandb.Image(fig)}, commit=True)
        plt.close('all')

    def _visualize_and_log_im(self, trajectory, state, p_traj_hat, a_traj_hat, window_traj_hat, name='recon'):
        """ Visualize and log the results.

        Args:
            state: The current training state.
        """

        p_traj_hat_flat = jnp.reshape(p_traj_hat, (-1, *p_traj_hat.shape[2:]))
        a_traj_hat_flat = jnp.reshape(a_traj_hat, (-1, *a_traj_hat.shape[2:]))
        window_traj_hat_flat = jnp.reshape(window_traj_hat, (-1, *window_traj_hat.shape[2:]))

        # Broadcast coords over batch dimension
        coords = jnp.broadcast_to(self.coords, (p_traj_hat_flat.shape[0], *self.coords.shape))

        out_all = None
        # Loop over all coords in chunks of self.config.training.max_num_sampled_pixels
        for i in range(0, coords.shape[1], self.config.training.max_num_sampled_points // 4):
            out = self.apply_nef_jitted(state.params['nef'],
                                        coords[:, i:i + self.config.training.max_num_sampled_points // 4], p_traj_hat_flat,
                                        a_traj_hat_flat, window_traj_hat_flat)

            if i == 0:
                out_all = out
            else:
                out_all = jnp.concatenate((out_all, out), axis=1)

        # Unflatten time and batch dimension
        out_all = out_all.reshape(*trajectory.shape)

        # Select first channel
        if trajectory.shape[-1] > 1:
            trajectory = trajectory[:, :, :, :, 1:]
            out_all = out_all[:, :, :, :, 1:]

        # Plot the first n images
        for i, img in enumerate(out_all):
            if i >= self.num_logged_samples:
                break

            fig, ax = plt.subplots(4, trajectory.shape[1], figsize=(60, 12))
            # Get min and max for diff plot
            vmin = 0
            vmax = np.max(trajectory[i]) - np.min(trajectory[i])

            for t in range(trajectory.shape[1]):
                ax[0, t].imshow(np.clip(trajectory[i, t], 0, 1), cmap='coolwarm')
                ax[0, t].axis('off')
                ax[0, t].set_title(f'T={t} / {np.mean((trajectory[i, t] - out_all[i, t])**2):.2E} ')
                ax[1, t].imshow(np.clip(out_all[i, t], 0, 1), cmap='coolwarm')
                ax[1, t].axis('off')

                # Plot abs diff
                ax[2, t].imshow(np.abs(out_all[i, t] - trajectory[i, t]), cmap='Reds', vmin=vmin, vmax=vmax)
                ax[2, t].axis('off')

                ax[3, t].imshow(np.abs(out_all[i, t] - trajectory[i, t]), cmap='Reds', vmin=vmin, vmax=vmax)

                # Plot poses
                y = (p_traj_hat[i, t, :, 0] + 1) * self.config.dataset.image_shape[1] / 2
                x = (p_traj_hat[i, t, :, 1] + 1) * self.config.dataset.image_shape[2] / 2
                if p_traj_hat.shape[-1] > 2:
                    dy = jnp.cos(p_traj_hat[i, t, :, 2])
                    dx = jnp.sin(p_traj_hat[i, t, :, 2])
                    ax[3, t].quiver(x, y, dx, dy, angles='uv', scale_units='xy', color='b')
                ax[3, t].scatter(x, y, c='b')
                ax[3, t].axis('off')

            # Disable axis
            wandb.log({f'{name}{i}': wandb.Image(fig)}, commit=True)

        # plot the first n images  on the sphere
        if self.config.dataset.name in ["diff_sphere", "shallow_water", "shallow_water_low_res"]:

            # Generate data
            lats = np.linspace(0., np.pi, trajectory.shape[3])
            longs = np.linspace(0, 2 * np.pi, trajectory.shape[2])
            longs, lats = np.meshgrid(longs, lats, indexing='ij')

            # Convert spherical coordinates to Cartesian coordinates for the plot
            x = np.sin(lats) * np.cos(longs)
            y = np.sin(lats) * np.sin(longs)
            z = np.cos(lats)

            # shallow water we only print the top half
            if self.config.dataset.name in ["shallow_water", "shallow_water_low_res"]:
                x = x[:, z.shape[1]//2:]
                y = y[:, z.shape[1]//2:]
                trajectory = trajectory[:, :, :, z.shape[1]//2:]
                out_all = out_all[:, :, :, z.shape[1]//2:]
                z = z[:, z.shape[1]//2:]

            for i, img in enumerate(out_all):
                if i >= self.num_logged_samples:
                    break

                fig = plt.figure(figsize=(25, 10))

                # Get min and max for diff plot
                vmin = 0
                vmax = np.max(trajectory[i]) - np.min(trajectory[i])

                for t in range(trajectory.shape[1] // 2):
                    t = 2 * t
                    if self.config.dataset.name in ["shallow_water", "shallow_water_low_res"]:
                        cmap_gt = plt.cm.Blues((trajectory[i, t] - trajectory[i, t].min()) / (
                                    trajectory[i, t].max() - trajectory[i, t].min()))
                        cmap_re = plt.cm.Blues(
                            (out_all[i, t] - out_all[i, t].min()) / (out_all[i, t].max() - out_all[i, t].min()))
                    else:
                        cmap_gt = plt.cm.magma((trajectory[i, t] - trajectory[i, t].min()) / (
                                    trajectory[i, t].max() - trajectory[i, t].min()))
                        cmap_re = plt.cm.magma(
                            (out_all[i, t] - out_all[i, t].min()) / (out_all[i, t].max() - out_all[i, t].min()))
                    ax0 = fig.add_subplot(2, trajectory.shape[1]//2, 1 + t//2 , projection='3d')
                    ax1 = fig.add_subplot(2, trajectory.shape[1]//2, 1 + t//2 + trajectory.shape[1]//2, projection='3d')

                    ax0.plot_surface(x, y, z, facecolors=cmap_gt, rstride=1, cstride=1, shade=False)
                    ax0.axis('off')
                    ax0.set_title(f'T={t} / {np.mean((trajectory[i, t] - out_all[i, t])**2):.2E} ')
                    
                    ax1.plot_surface(x, y, z, facecolors=cmap_re, rstride=1, cstride=1, shade=False)

                    # Hide grid lines
                    ax0.grid(False)
                    ax1.grid(False)

                    # Hide axes
                    ax0.axis('off')
                    ax1.axis('off')

                    # Set viewpoint
                    ax0.view_init(elev=-45, azim=45, roll=180)
                    ax1.view_init(elev=-45, azim=45, roll=180)

                    if self.config.dataset.name in ["shallow_water", "shallow_water_low_res"]:
                        ax0.axes.set_xlim3d(left=-1, right=1)
                        ax0.axes.set_ylim3d(bottom=-1, top=1)
                        ax0.axes.set_zlim3d(bottom=-1, top=0)
                        ax1.axes.set_xlim3d(left=-1, right=1)
                        ax1.axes.set_ylim3d(bottom=-1, top=1)
                        ax1.axes.set_zlim3d(bottom=-1, top=0)

                        # Set equal scaling
                        ax0.set_box_aspect([1, 1, 0.5])
                        ax1.set_box_aspect([1, 1, 0.5])

                    else:
                        # Set equal scaling
                        ax0.set_box_aspect([1, 1, 1])
                        ax1.set_box_aspect([1, 1, 1])

                plt.tight_layout()
                # Disable axis
                wandb.log({f'sphere-{name}{i}': wandb.Image(fig)}, commit=True)

        plt.close('all')

    def transform_visualize_and_log_2d(self, trajectory, state, p_traj_hat, a_traj_hat, window_traj_hat, name='recon'):
        """ Apply transformations to the latents, get visualizations and log results. This is useful for sanity checking
        equivariance properties.

        Args:
            state: The current training state.
        """
        self.visualize_and_log(trajectory, state, p_traj_hat, a_traj_hat, window_traj_hat,
                               name=f"equiv-test/org-{name}")

        # Perform translation
        p_traj_hat_translated = p_traj_hat + 0.5
        self.visualize_and_log(trajectory, state, p_traj_hat_translated, a_traj_hat, window_traj_hat,
                               name=f"equiv-test/transl-{name}")

        # Perform rotation, create rot matrix for 35 deg rot
        rot_matrix = jnp.array([[np.cos(np.pi/6), -np.sin(np.pi/6)], [np.sin(np.pi/6), np.cos(np.pi/6)]])

        # Apply to positions
        if p_traj_hat.shape[3] == 2:
            p_traj_hat_rotated = jnp.einsum('btni,io->btno', p_traj_hat, rot_matrix)
        else:
            p_traj_hat_rotated = jnp.einsum('btni,io->btno', p_traj_hat[:, :, :, :2], rot_matrix)
            p_traj_hat_rotated = jnp.concatenate((p_traj_hat_rotated, p_traj_hat[:, :, :, 2:] - np.pi/6), axis=3)

        self.visualize_and_log(trajectory, state, p_traj_hat_rotated, a_traj_hat, window_traj_hat,
                               name=f"equiv-test/rot-{name}")

    # Define the following methods in the derived classes
    def init_train_state(self):
        raise NotImplementedError("init_train_state method not implemented, please implement in derived class")

    def train_step(self, state, batch):
        raise NotImplementedError("train_step method not implemented, please implement in derived class")

    def val_step(self, state, batch):
        raise NotImplementedError("val_step method not implemented, please implement in derived class")

    def create_functions(self):
        raise NotImplementedError("create_functions method not implemented, please implement in derived class")



