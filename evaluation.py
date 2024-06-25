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

"""Utility functions for computing FID/Inception scores."""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
import six
import os

INCEPTION_V3 = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
INCEPTION_V1 = 'path_to_inception_v1_model'  # Update this with the actual path or URL
DEFAULT_DTYPES = {
    'logits': torch.float32,
    'pool_3': torch.float32
}
INCEPTION_DEFAULT_IMAGE_SIZE = 299

def get_inception_model(inceptionv3=False):
    if inceptionv3:
        model = models.inception_v3(pretrained=False)
        model.load_state_dict(torch.hub.load_state_dict_from_url(INCEPTION_V3))
        model.fc = nn.Identity()  # Remove the final fully connected layer
        return model
    else:
        # Load InceptionV1 from local or custom URL
        model = models.inception_v3(pretrained=False)  # Placeholder
        model.load_state_dict(torch.load(INCEPTION_V1))
        model.fc = nn.Identity()
        return model

def load_dataset_stats(config):
    """Load the pre-computed dataset statistics."""
    if config.data.dataset == 'CIFAR10':
        filename = 'assets/stats/cifar10_stats.npz'
    elif config.data.dataset == 'CELEBA':
        filename = 'assets/stats/celeba_stats.npz'
    elif config.data.dataset == 'LSUN':
        filename = f'assets/stats/lsun_{config.data.category}_{config.data.image_size}_stats.npz'
    else:
        raise ValueError(f'Dataset {config.data.dataset} stats not found.')

    with open(filename, 'rb') as fin:
        stats = np.load(fin)
        return stats

def classifier_fn_from_torchhub(output_fields, inception_model, return_tensor=False):
    """Returns a function that can be used as a classifier function.

    Args:
        output_fields: A string, list, or `None`. If present, assume the module
            outputs a dictionary, and select this field.
        inception_model: A model loaded from PyTorch hub.
        return_tensor: If `True`, return a single tensor instead of a dictionary.

    Returns:
        A one-argument function that takes an image Tensor and returns outputs.
    """
    if isinstance(output_fields, six.string_types):
        output_fields = [output_fields]

    def _classifier_fn(images):
        with torch.no_grad():
            output = inception_model(images)
        if output_fields is not None:
            output = {x: output[x] for x in output_fields}
        if return_tensor:
            assert len(output) == 1
            output = list(output.values())[0]
        return {k: v.flatten(start_dim=1) for k, v in output.items()}

    return _classifier_fn

def run_inception_jit(inputs, inception_model, num_batches=1, inceptionv3=False):
    """Running the inception network. Assuming input is within [0, 255]."""
    if not inceptionv3:
        inputs = (inputs.float() - 127.5) / 127.5
    else:
        inputs = inputs.float() / 255.

    results = []
    for i in range(num_batches):
        batch = inputs[i * (inputs.size(0) // num_batches): (i + 1) * (inputs.size(0) // num_batches)]
        results.append(classifier_fn_from_torchhub(None, inception_model)(batch))
    return {k: torch.cat([result[k] for result in results], dim=0) for k in results[0].keys()}

def run_inception_distributed(input_tensor, inception_model, num_batches=1, inceptionv3=False):
    """Distribute the inception network computation to all available GPUs.

    Args:
        input_tensor: The input images. Assumed to be within [0, 255].
        inception_model: The inception network model obtained from torch hub.
        num_batches: The number of batches used for dividing the input.
        inceptionv3: If `True`, use InceptionV3, otherwise use InceptionV1.

    Returns:
        A dictionary with key `pool_3` and `logits`, representing the pool_3 and
            logits of the inception network respectively.
    """
    num_gpus = torch.cuda.device_count()
    input_tensors = torch.split(input_tensor, input_tensor.size(0) // num_gpus, dim=0)
    pool3 = []
    logits = [] if not inceptionv3 else None

    for i, tensor in enumerate(input_tensors):
        device = torch.device(f'cuda:{i}')
        inception_model.to(device)
        tensor = tensor.to(device)
        with torch.no_grad():
            res = run_inception_jit(tensor, inception_model, num_batches=num_batches, inceptionv3=inceptionv3)
            if not inceptionv3:
                pool3.append(res['pool_3'])
                logits.append(res['logits'])  # pytype: disable=attribute-error
            else:
                pool3.append(res)

    return {
        'pool_3': torch.cat(pool3, dim=0).cpu(),
        'logits': torch.cat(logits, dim=0).cpu() if not inceptionv3 else None
    }
