# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import os
import logging
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import evaluation
import likelihood
import sde_lib
import numpy as np

def get_config_from_file(config_file):
    print(f'config_file:{config_file}')
    import importlib
    # Implement this function to read the config from a file
    # and return it as a dictionary or appropriate object

    # Get the absolute path of the config file
    config_file_path = os.path.abspath(config_file)

    # Create a module name from the file name
    module_name = os.path.basename(config_file_path).split('.')[0]

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, config_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Assuming the module defines a function `get_config()` that returns the config object
    if hasattr(module, 'get_config'):
        config = module.get_config()  # Call the function to get the config object
        return config
    else:
        raise AttributeError(f"The module {module_name} does not define 'get_config()' function.")

def resize_tensor(tensor, width2, height2, interpolation="bilinear"):
    """
    Resizes a PyTorch tensor of size (batch, channels, width, height) to (batch, channels, width2, height2) using the specified interpolation.

    Args:
        tensor: The input tensor to resize.
        width2: The target width after resizing.
        height2: The target height after resizing.
        interpolation: The interpolation mode to use. Supported options are "nearest", "bilinear", "bicubic", "trilinear" (for 3D tensors), and "lanczos" (if available in your PyTorch installation).

    Returns:
        A resized tensor with the specified dimensions and interpolation mode.
    """

    if interpolation not in ["nearest", "bilinear", "bicubic", "trilinear", "lanczos"]:
        raise ValueError(f"Unsupported interpolation mode: {interpolation}")

    # Use nn.functional.interpolate for flexibility and efficiency
    resized_tensor = torch.nn.functional.interpolate(tensor, size=(height2, width2), mode=interpolation)
    return resized_tensor

def train(config, workdir):
    """Runs the training pipeline.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    tb_dir = os.path.join(workdir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

    # Build data iterators
    train_ds, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)
    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting)

    # Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (config.training.batch_size, config.data.num_channels,
                          config.data.image_size, config.data.image_size)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    num_train_steps = config.training.n_iters

    logging.info("Starting training loop at step %d." % (initial_step,))

    for step in range(initial_step, num_train_steps + 1):
        try:
            batch = next(train_iter)[0].to(config.device).float()
        except StopIteration:
            train_iter = iter(train_ds)
            batch = next(train_iter)[0].to(config.device).float()
        print(f'batch0[{step} {initial_step}/{num_train_steps}]:{batch.shape} eval_freq:{config.training.eval_freq} snapshot_freq:{config.training.snapshot_freq}')
        #batch = resize_tensor(batch, 128, 128)  # Fix
        #batch = batch.permute(0, 3, 1, 2)
        #batch = batch.permute(0, 3, 2, 1)
        #print(f'batch1:{batch.shape}')
        batch = scaler(batch)
        #print(f'batch2:{batch.shape}')
        loss = train_step_fn(state, batch)
        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
            writer.add_scalar("training_loss", loss.item(), step)

        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        if step % config.training.eval_freq == 0:
            try:
                eval_batch = next(eval_iter)[0].to(config.device).float()
            except StopIteration:
                eval_iter = iter(eval_ds)
                eval_batch = next(eval_iter)[0].to(config.device).float()
            #eval_batch = resize_tensor(eval_batch, 128, 128)   # Fix
            #eval_batch = eval_batch.permute(0, 3, 1, 2)
            eval_batch = scaler(eval_batch)
            eval_loss = eval_step_fn(state, eval_batch)
            logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
            writer.add_scalar("eval_loss", eval_loss.item(), step)

        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            save_step = step // config.training.snapshot_freq
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

            if config.training.snapshot_sampling:
                print('a0')
                ema.store(score_model.parameters())
                print('a1')
                ema.copy_to(score_model.parameters())
                print('a2')
                sample, n = sampling_fn(score_model)
                print('a3')
                ema.restore(score_model.parameters())
                print('a4')
                this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                print('a5')
                os.makedirs(this_sample_dir, exist_ok=True)
                print('a6')
                nrow = int(np.sqrt(sample.shape[0]))
                print('a7')
                image_grid = make_grid(sample, nrow, padding=2)
                print('a8')
                sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                print('a9')
                with open(os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
                    np.save(fout, sample)
                print('a10')
                with open(os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
                    save_image(image_grid, fout)
                print('a11')



    # for step in range(initial_step, num_train_steps + 1):
    #     print(f'next(train_iter):{next(train_iter)}')
    #     batch = next(train_iter)['image'].to(config.device).float()
    #     batch = batch.permute(0, 3, 1, 2)
    #     batch = scaler(batch)
    #     loss = train_step_fn(state, batch)
    #     if step % config.training.log_freq == 0:
    #         logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
    #         writer.add_scalar("training_loss", loss.item(), step)

    #     if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
    #         save_checkpoint(checkpoint_meta_dir, state)

    #     if step % config.training.eval_freq == 0:
    #         eval_batch = next(eval_iter)['image'].to(config.device).float()
    #         eval_batch = eval_batch.permute(0, 3, 1, 2)
    #         eval_batch = scaler(eval_batch)
    #         eval_loss = eval_step_fn(state, eval_batch)
    #         logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
    #         writer.add_scalar("eval_loss", eval_loss.item(), step)

    #     if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
    #         save_step = step // config.training.snapshot_freq
    #         save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

    #         if config.training.snapshot_sampling:
    #             ema.store(score_model.parameters())
    #             ema.copy_to(score_model.parameters())
    #             sample, n = sampling_fn(score_model)
    #             ema.restore(score_model.parameters())
    #             this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
    #             os.makedirs(this_sample_dir, exist_ok=True)
    #             nrow = int(np.sqrt(sample.shape[0]))
    #             image_grid = make_grid(sample, nrow, padding=2)
    #             sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
    #             with open(os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
    #                 np.save(fout, sample)
    #             with open(os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
    #                 save_image(image_grid, fout)

def evaluate(config, workdir, eval_folder="eval"):
    """Evaluate trained models.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints.
        eval_folder: The subfolder for storing evaluation results. Default to
        "eval".
    """
    eval_dir = os.path.join(workdir, eval_folder)
    os.makedirs(eval_dir, exist_ok=True)

    train_ds, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization, evaluation=True)
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")

    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    state = restore_checkpoint(os.path.join(checkpoint_dir, "checkpoint.pth"), state, config.device)
    ema.copy_to(score_model.parameters())

    if config.eval.enable_bpd:
        eval_ds_bpd, _ = datasets.get_dataset(config, uniform_dequantization=False, evaluation=True)
        bpds = likelihood.get_bpd_dataset(config, eval_ds_bpd, scaler, state, inverse_scaler, sde)
        np.save(os.path.join(eval_dir, "bpds.npy"), bpds)
        logging.info("bps: %.5e" % bpds.mean().item())

    if config.eval.enable_sampling:
        sampling_shape = (config.eval.batch_size, config.data.num_channels,
                          config.data.image_size, config.data.image_size)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        sample, n = sampling_fn(score_model)
        ema.restore(score_model.parameters())
        sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        with open(os.path.join(eval_dir, "sample.npy"), "wb") as fout:
            np.save(fout, sample)
        with open(os.path.join(eval_dir, "sample.png"), "wb") as fout:
            save_image(make_grid(sample, nrow=int(np.sqrt(config.eval.batch_size)), padding=2), fout)

