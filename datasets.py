"""Return training and evaluation/test datasets from config files."""
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, SVHN, CelebA, LSUN #, FFHQ
from torchvision.io import read_image
import torchvision.transforms.functional as TF

def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x

def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x

def crop_resize(image, resolution):
    """Crop and resize an image to the given resolution."""
    h, w = image.shape[-2], image.shape[-1]
    crop = min(h, w)
    top = (h - crop) // 2
    left = (w - crop) // 2
    image = TF.crop(image, top, left, crop, crop)
    image = TF.resize(image, [resolution, resolution], interpolation=TF.InterpolationMode.BICUBIC)
    return image

def resize_small(image, resolution):
    """Shrink an image to the given resolution."""
    h, w = image.shape[-2], image.shape[-1]
    ratio = resolution / min(h, w)
    new_size = (int(h * ratio), int(w * ratio))
    return TF.resize(image, new_size, interpolation=TF.InterpolationMode.BICUBIC)

def central_crop(image, size):
    """Crop the center of an image to the given size."""
    h, w = image.shape[-2], image.shape[-1]
    top = (h - size) // 2
    left = (w - size) // 2
    return TF.crop(image, top, left, size, size)

def get_dataset(config, uniform_dequantization=False, evaluation=False):
    """Create data loaders for training and evaluation.

    Args:
        config: A ml_collection.ConfigDict parsed from config files.
        uniform_dequantization: If `True`, add uniform dequantization to images.
        evaluation: If `True`, fix number of epochs to 1.

    Returns:
        train_ds, eval_ds, dataset_builder.
    """
    # Compute batch size for this worker.
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size

    if batch_size % torch.cuda.device_count() != 0:
        raise ValueError(f'Batch sizes ({batch_size}) must be divisible by the number of devices ({torch.cuda.device_count()})')

    # Reduce this when image resolution is too large and data pointer is stored
    shuffle_buffer_size = 10000
    num_epochs = None if not evaluation else 1

    # Define the transformation for each dataset
    if config.data.dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((config.data.image_size, config.data.image_size), interpolation=transforms.InterpolationMode.BICUBIC)
        ])
        dataset_builder = CIFAR10

    elif config.data.dataset == 'SVHN':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((config.data.image_size, config.data.image_size), interpolation=transforms.InterpolationMode.BICUBIC)
        ])
        dataset_builder = SVHN

    elif config.data.dataset == 'CELEBA':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(140),
            transforms.Resize(config.data.image_size, interpolation=transforms.InterpolationMode.BICUBIC)
        ])
        dataset_builder = CelebA

    elif config.data.dataset == 'LSUN':
        if config.data.image_size == 128:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(config.data.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(config.data.image_size)
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda img: crop_resize(img, config.data.image_size))
            ])
        dataset_builder = LSUN

    elif config.data.dataset in ['FFHQ', 'CelebAHQ']:
        # Custom dataset class to handle FFHQ and CelebAHQ dataset from TFRecord
        class CustomDataset(Dataset):
            def __init__(self, tfrecords_path, transform=None):
                self.tfrecords_path = tfrecords_path
                self.transform = transform
                self.samples = self.load_samples()

            def load_samples(self):
                # Implement loading of TFRecord samples
                pass

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                sample = self.samples[idx]
                image = read_image(sample['image_path'])
                if self.transform:
                    image = self.transform(image)
                return image, None

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip() if config.data.random_flip and not evaluation else transforms.Lambda(lambda x: x)
        ])
        dataset_builder = CustomDataset
    else:
        raise NotImplementedError(f'Dataset {config.data.dataset} not yet supported.')

    # Customize preprocess functions for each dataset.
    if config.data.dataset in ['FFHQ', 'CelebAHQ']:
        def preprocess_fn(sample):
            image = sample
            if config.data.random_flip and not evaluation:
                image = TF.hflip(image)
            if uniform_dequantization:
                image = (torch.rand_like(image) + image * 255.) / 256.
            return image

    else:
        def preprocess_fn(sample):
            """Basic preprocessing function scales data to [0, 1) and randomly flips."""
            image, label = sample
            if config.data.random_flip and not evaluation:
                image = TF.hflip(image)
            if uniform_dequantization:
                image = (torch.rand_like(image) + image * 255.) / 256.
            return image, label

    def create_dataset(dataset_builder, split):
        print(f'create_dataset dataset_builder:{dataset_builder}')
        if isinstance(dataset_builder, Dataset):
            dataset = dataset_builder(root=config.data.path, split=split, transform=transform, download=True)
        else:
            #dataset = dataset_builder(tfrecords_path=config.data.tfrecords_path, transform=transform)
            dataset = dataset_builder(root=config.data.path, download=True, transform=transform)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        return dataloader

    train_ds = create_dataset(dataset_builder, 'train')
    eval_ds = create_dataset(dataset_builder, 'test' if config.data.dataset != 'CELEBA' else 'valid')
    return train_ds, eval_ds, dataset_builder
