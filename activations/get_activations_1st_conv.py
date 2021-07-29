"""Pre-train descendant for source dataset."""
import torch.nn as nn
import torch.optim as optim

import params
from utils import make_variable, save_model

import os
import numpy as np
from sklearn.metrics import accuracy_score

def apply_descendant(descendant, data_loader):
    """Evaluate descendant for source domain."""
    # set eval state for Dropout and BN layers
    descendant.eval()
    descendant.cuda()

    activations = []
    ys = []

    # init loss and accuracy
    loss = 0.0
    acc = 0.0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()

        preds = descendant(images)

        for pred, label in zip(preds, labels):
            activations.append(pred.detach().cpu().numpy())
            ys.append(label.detach().cpu().numpy())

    activations = np.asarray(activations)
    ys = ys.asarray(activations)

    print('the activations after the 1st conv have shape {}'.format(activations.shape))
    np.save('snapshots//1st_conv_activations.npy', activations)

    print('the activations after the 1st conv have labels with shape {}'.format(ys.shape))
    np.save('snapshots//1st_conv_activations_labels.npy', activations)

    return activations, ys
