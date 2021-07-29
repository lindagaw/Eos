"""Pre-train successor for source dataset."""
import torch.nn as nn
import torch.optim as optim

import params
from utils import make_variable, save_model

import os
import numpy as np
from sklearn.metrics import accuracy_score

def apply_successor(successor, data_loader):
    """Evaluate successor for source domain."""
    # set eval state for Dropout and BN layers
    successor.eval()
    successor.cuda()

    activations = []

    # init loss and accuracy
    loss = 0.0
    acc = 0.0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()

        preds = successor(images)

        for pred in preds:
            activations.append(pred.detach().cpu().numpy())

    activations = np.asarray(activations)
    print(activations.shape)
    np.save('snapshots//2nd_conv_activations.npy', activations)

    return activations
