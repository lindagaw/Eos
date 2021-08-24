"""Dataset setting and data loader for SVHN."""


import torch
from torchvision import datasets, transforms

import params


def get_cifar_10(train):
    """Get SVHN dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=params.dataset_mean,
                                          std=params.dataset_std)])
    pre_process =  transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((224,224)),
                                        transforms.Normalize((0.5,), (0.5,))])
    # dataset and data loader
    cifar_10_dataset = datasets.CIFAR10(root=params.data_root,
                                   #train=train,
                                   transform=pre_process,
                                   download=True)

    cifar_10_data_loader = torch.utils.data.DataLoader(
        dataset=cifar_10_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return cifar_10_data_loader
