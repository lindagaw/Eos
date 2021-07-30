"""Test script to classify target data."""

import torch
import torch.nn as nn

from utils import make_variable
from sklearn.metrics import accuracy_score

import numpy as np

def eval_tgt_with_probe(encoder, critic, src_classifier, tgt_classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    src_classifier.eval()
    tgt_classifier.eval()
    # init loss and accuracy
    loss = 0
    acc = 0
    f1 = 0

    ys_pred = []
    ys_true = []
    # set loss function
    criterion = nn.CrossEntropyLoss()
    flag = False
    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()

        probeds = critic(encoder(images))

        for image, label, probed in zip(images, labels, probeds):
            if torch.argmax(probed) == 1:
                pred = torch.argmax(src_classifier(encoder(torch.unsqueeze(image, 0)))).detach().cpu().numpy()
            else:
                pred = torch.argmax(tgt_classifier(encoder(torch.unsqueeze(image, 0)))).detach().cpu().numpy()

        ys_pred.append(np.squeeze(pred))
        ys_true.append(np.squeeze(label.detach().cpu().numpy()))

    acc = accuracy_score(ys_true, ys_pred)


    print("Avg Loss = {}, Accuracy = {:2%}".format(loss, acc))

def eval_tgt(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0.0
    acc = 0.0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()

        preds = classifier(encoder(images))
        loss += criterion(preds, labels).data

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
