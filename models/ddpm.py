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
"""DDPM model.

This code is the pytorch equivalent of:
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py
"""
import torch
import torch.nn as nn
import functools

from . import utils, layers, normalization

RefineBlock = layers.RefineBlock
ResidualBlock = layers.ResidualBlock
ResnetBlockDDPM = layers.ResnetBlockDDPM
Upsample = layers.Upsample
Downsample = layers.Downsample
conv3x3 = layers.ddpm_conv3x3
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@utils.register_model(name='ddpm')
class DDPM(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.act = act = get_act(config)
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))

    self.nf = nf = config.model.nf
    ch_mult = config.model.ch_mult
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    resamp_with_conv = config.model.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

    AttnBlock = functools.partial(layers.AttnBlock)
    self.conditional = conditional = config.model.conditional
    ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
    if conditional:
      # Condition on noise levels.
      modules = [nn.Linear(nf, nf * 4)]
      modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
      nn.init.zeros_(modules[0].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
      nn.init.zeros_(modules[1].bias)

    self.centered = config.data.centered
    channels = config.data.num_channels

    # Downsampling block
    modules.append(conv3x3(channels, nf))
    hs_c = [nf]
    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch
        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
        hs_c.append(in_ch)
      if i_level != num_resolutions - 1:
        modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))

    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        h0 = in_ch
        h1 = hs_c.pop()
        #modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
        modules.append(ResnetBlock(in_ch=h0 + h1, out_ch=out_ch))
        in_ch = out_ch
      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))
      if i_level != 0:
        modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))

    assert not hs_c
    modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
    modules.append(conv3x3(in_ch, channels, init_scale=0.))
    self.all_modules = nn.ModuleList(modules)

    self.scale_by_sigma = config.model.scale_by_sigma

  # def __init__(self, config):
  #     super().__init__()
  #     self.act = act = get_act(config)
  #     self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))

  #     self.nf = nf = config.model.nf // 2  # Reduce the number of filters by half
  #     ch_mult = config.model.ch_mult
  #     self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
  #     self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
  #     dropout = config.model.dropout
  #     resamp_with_conv = config.model.resamp_with_conv
  #     self.num_resolutions = num_resolutions = len(ch_mult)
  #     self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

  #     AttnBlock = functools.partial(layers.AttnBlock)
  #     self.conditional = conditional = config.model.conditional
  #     ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
  #     if conditional:
  #         # Condition on noise levels.
  #         modules = [nn.Linear(nf, nf * 4)]
  #         modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
  #         nn.init.zeros_(modules[0].bias)
  #         modules.append(nn.Linear(nf * 4, nf * 4))
  #         modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
  #         nn.init.zeros_(modules[1].bias)

  #     self.centered = config.data.centered
  #     channels = config.data.num_channels

  #     # Downsampling block
  #     modules.append(conv3x3(channels, nf))
  #     hs_c = [nf]
  #     in_ch = nf
  #     for i_level in range(num_resolutions):
  #         # Residual blocks for this resolution
  #         for i_block in range(num_res_blocks):
  #             out_ch = nf * ch_mult[i_level]
  #             modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
  #             in_ch = out_ch
  #             if all_resolutions[i_level] in attn_resolutions:
  #                 modules.append(AttnBlock(channels=in_ch))
  #             hs_c.append(in_ch)
  #         if i_level != num_resolutions - 1:
  #             modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
  #             hs_c.append(in_ch)

  #     in_ch = hs_c[-1]
  #     modules.append(ResnetBlock(in_ch=in_ch))
  #     modules.append(AttnBlock(channels=in_ch))
  #     modules.append(ResnetBlock(in_ch=in_ch))

  #     # Upsampling block
  #     for i_level in reversed(range(num_resolutions)):
  #         for i_block in range(num_res_blocks + 1):
  #             out_ch = nf * ch_mult[i_level]
  #             h0 = in_ch
  #             h1 = hs_c.pop()
  #             modules.append(ResnetBlock(in_ch=h0 + h1, out_ch=out_ch))
  #             in_ch = out_ch
  #         if all_resolutions[i_level] in attn_resolutions:
  #             modules.append(AttnBlock(channels=in_ch))
  #         if i_level != 0:
  #             modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))

  #     assert not hs_c
  #     modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
  #     modules.append(conv3x3(in_ch, channels, init_scale=0.))
  #     self.all_modules = nn.ModuleList(modules)

  #     self.scale_by_sigma = config.model.scale_by_sigma
      
  def forward(self, x, labels):
    modules = self.all_modules
    m_idx = 0
    if self.conditional:
      # timestep/scale embedding
      timesteps = labels
      temb = layers.get_timestep_embedding(timesteps, self.nf)
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None

    if self.centered:
      # Input is in [-1, 1]
      h = x
    else:
      # Input is in [0, 1]
      h = 2 * x - 1.

    # Downsampling block
    hs = [modules[m_idx](h)]
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        #print(f'm_idx:{m_idx} i_block:{i_block} hs[-1]:{hs[-1].shape}')
        h = modules[m_idx](hs[-1], temb)
        m_idx += 1
        #if h.shape[-1] in self.attn_resolutions:
        if self.all_resolutions[i_level] in self.attn_resolutions:
          h = modules[m_idx](h)
          m_idx += 1
        hs.append(h)
      if i_level != self.num_resolutions - 1:
        hs.append(modules[m_idx](hs[-1]))
        m_idx += 1

    h = hs[-1]
    h = modules[m_idx](h, temb)
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    h = modules[m_idx](h, temb)
    m_idx += 1

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        #print(f'm_idx:{m_idx} i_block:{i_block} h:{h.shape}')
        h0 = h
        h1 = hs.pop()
        #h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
        h = modules[m_idx](torch.cat([h0, h1], dim=1), temb)
        m_idx += 1
      #if h.shape[-1] in self.attn_resolutions:
      if self.all_resolutions[i_level] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1
      if i_level != 0:
        h = modules[m_idx](h)
        m_idx += 1

    assert not hs
    h = self.act(modules[m_idx](h))
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    assert m_idx == len(modules)

    if self.scale_by_sigma:
      # Divide the output by sigmas. Useful for training with the NCSN loss.
      # The DDPM loss scales the network output by sigma in the loss function,
      # so no need of doing it here.
      used_sigmas = self.sigmas[labels, None, None, None]
      h = h / used_sigmas

    return h

  def noise_images(self, x, t):
      """
      Adds noise to the images `x` according to the time step `t`.
      """
      t = t.long()  # Ensure t is of long type
      noise = torch.randn_like(x).to(x.device)
      sigmas_t = self.sigmas[t].view(-1, 1, 1, 1).to(x.device)
      noised_images = x + sigmas_t * noise
      return noised_images, noise

  def sample_timesteps(self, n):
      """
      Samples timesteps uniformly between 0 and len(sigmas) - 1.
      """
      return torch.randint(0, len(self.sigmas), (n,))

  def sample(self, n):
      """
      Generates `n` samples using the reverse diffusion process.
      """
      device = next(self.parameters()).device
      shape = (n, 3, 64, 64)  # Assuming 3 channels and 64x64 images
      x = torch.randn(shape).to(device)
      for i in reversed(range(len(self.sigmas))):
          t = torch.tensor([i] * n, device=device)
          x = self.forward(x, t)
      return x


def main():

    from torch.cuda.amp import autocast, GradScaler
    import os
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

    config = get_config_from_file('configs/subvp/cifar10_ddpm_continuous.py')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 64
    batch_size = 2  # Reduced batch size
    noise_steps = 1000
    beta_start = 1e-4
    beta_end = 0.02
    
    model = DDPM(config=config)
    model = model.to(device)
    
    # Generate dummy data
    dummy_images = torch.randn((batch_size, 3, img_size, img_size)).to(device)
    dummy_timesteps = torch.randint(0, len(model.sigmas), (batch_size,)).to(device)  # Ensure dummy_timesteps are integers
    
    scaler = GradScaler()
    
    with autocast():
        # Test forward pass
        output = model(dummy_images, dummy_timesteps)
        print(f"Output shape: {output.shape}")
        
        # Test noise_images function
        noised_images, epsilon = model.noise_images(dummy_images, dummy_timesteps)
        print(f"Noised images shape: {noised_images.shape}")
        print(f"Epsilon shape: {epsilon.shape}")
        
        # Test sampling
        sampled_images = model.sample(n=batch_size)
        print(f"Sampled images shape: {sampled_images.shape}")

if __name__ == "__main__":
    main()