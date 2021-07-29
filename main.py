"""Main script for ADDA."""

import params
from core import eval_src, eval_tgt, train_src, train_tgt, train_tgt_classifier
from core import train_progenitor, eval_progenitor

from activations import apply_descendant

from models import Discriminator, LeNetClassifier, LeNetEncoder
from models import Progenitor, Descendant, Successor
from utils import get_data_loader, init_model, init_random_seed, load_chopped_state_dict

if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset
    src_data_loader = get_data_loader(params.src_dataset)
    src_data_loader_eval = get_data_loader(params.src_dataset, train=False)
    tgt_data_loader = get_data_loader(params.tgt_dataset)
    tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)

    # train the original source classifier, the Progenitor
    print(">>> the original source classifier, the Progenitor <<<")
    progenitor = init_model(net=Progenitor(),
                             restore=params.progenitor_restore)
    print(progenitor)
    progenitor = train_progenitor(progenitor, src_data_loader)
    eval_progenitor(progenitor, src_data_loader_eval)

    print(">>> load the chopped model with 1 conv, the Descendant <<<")
    descendant = load_chopped_state_dict(model=Descendant(), pretrained_dict=params.progenitor_restore)
    print(descendant)

    print(">>> load the chopped model with 2 convs, the Successor <<<")
    successor = load_chopped_state_dict(model=Successor(), pretrained_dict=params.progenitor_restore)
    print(successor)

    print(">>> get the activations after the 1st conv, using Descendant <<<")
    descendant_activations = apply_descendant(descendant, src_data_loader)

    print(">>> get the activations after the 2nd conv, using Successor <<<")
    successor_activations = apply_successor(successor, src_data_loader)

'''
    # load models
    src_encoder = init_model(net=LeNetEncoder(),
                             restore=params.src_encoder_restore)
    src_classifier = init_model(net=LeNetClassifier(),
                                restore=params.src_classifier_restore)
    tgt_classifier = init_model(net=LeNetClassifier(),
                                restore='')
    tgt_encoder = init_model(net=LeNetEncoder(),
                             restore=params.tgt_encoder_restore)
    critic = init_model(Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims),
                        restore=params.d_model_restore)

    # train source model
    print("=== Training classifier for source domain ===")
    print(">>> Source Encoder <<<")
    print(src_encoder)
    print(">>> Source Classifier <<<")
    print(src_classifier)

    if not (src_encoder.restored and src_classifier.restored and
            params.src_model_trained):
        src_encoder, src_classifier = train_src(
            src_encoder, src_classifier, src_data_loader)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_classifier, src_data_loader_eval)

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    print(">>> Target Encoder <<<")
    print(tgt_encoder)
    print(">>> Critic <<<")
    print(critic)

    # init weights of target encoder with those of source encoder
    if not tgt_encoder.restored:
        tgt_encoder.load_state_dict(src_encoder.state_dict())

    if not (tgt_encoder.restored and critic.restored and
            params.tgt_model_trained):
        tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic,
                                src_data_loader, tgt_data_loader)

    tgt_encoder, tgt_classifier = train_tgt_classifier(
        tgt_encoder, tgt_classifier, tgt_data_loader)


    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
    print(">>> domain adaption <<<")
    eval_tgt(tgt_encoder, tgt_classifier, tgt_data_loader_eval)
'''
