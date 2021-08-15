"""Main script for ADDA."""
import pretty_errors
import params

from torchvision import datasets, transforms, models

from core import eval_src, eval_tgt, train_src, train_tgt, train_tgt_classifier
from core import train_progenitor, eval_progenitor
from core import eval_tgt_with_probe
from core import get_distribution, eval_ADDA

from activations import apply_descendant, apply_successor

from models import Discriminator, LeNetClassifier, LeNetEncoder
from models import Progenitor, Descendant, Successor
from models import LeNet_Conv_1_Encoder, LeNet_Conv_1_Classifier, LeNet_Conv_2_Encoder, LeNet_Conv_2_Classifier

from utils import get_data_loader, init_model, init_random_seed, load_chopped_state_dict

from datasets import get_conv_1_activations, get_conv_2_activations
from datasets import get_office_home, get_office_31

import torch

import gc
gc.collect()
torch.cuda.empty_cache()

if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset
    #src_data_loader = get_data_loader(params.src_dataset)
    #src_data_loader_eval = get_data_loader(params.src_dataset, train=False)
    #tgt_data_loader = get_data_loader(params.tgt_dataset)
    #tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)

    src_data_loader = get_office_31(dataset = 'office-31-amazon', train=True)
    src_data_loader_eval = get_office_31(dataset = 'office-31-amazon', train=False)
    tgt_data_loader = get_office_31(dataset = 'office-31-webcam', train=True)
    tgt_data_loader_eval = get_office_31(dataset = 'office-31-webcam', train=False)

    model = model = models.resnet152(pretrained=True)
    newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
    print(newmodel)

'''
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

    print(">>> get the activations after the 1st conv, using Descendant <<<")

    apply_descendant(descendant, src_data_loader, 'src', 'dev')
    apply_descendant(descendant, src_data_loader_eval, 'src', 'eval')
    apply_descendant(descendant, tgt_data_loader, 'tgt', 'dev')
    apply_descendant(descendant, tgt_data_loader_eval, 'tgt', 'eval')

    print(">>> construct dataloader after activations from 1st conv <<<")
    src_conv_1_activations_data_loader = get_conv_1_activations(train=True, dataset='src')
    src_conv_1_activations_data_loader_eval = get_conv_1_activations(train=False, dataset='src')
    tgt_conv_1_activations_data_loader = get_conv_1_activations(train=True, dataset='tgt')
    tgt_conv_1_activations_data_loader_eval = get_conv_1_activations(train=False, dataset='tgt')



    print(">>> train the src_encoder, tgt_encoder, src_classifier, tgt_classifier <<<")

    # load models
    src_encoder = init_model(net=LeNet_Conv_1_Encoder(),
                             restore=params.src_encoder_restore)
    src_classifier = init_model(net=LeNet_Conv_1_Classifier(),
                                restore=params.src_classifier_restore)
    tgt_classifier = init_model(net=LeNet_Conv_1_Classifier(),
                                restore='')
    tgt_encoder = init_model(net=LeNet_Conv_1_Encoder(),
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
            src_encoder, src_classifier, src_conv_1_activations_data_loader)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_classifier, src_conv_1_activations_data_loader_eval)

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
                                src_conv_1_activations_data_loader, tgt_conv_1_activations_data_loader)

    tgt_encoder, tgt_classifier = train_tgt_classifier(
        tgt_encoder, tgt_classifier, tgt_conv_1_activations_data_loader)


    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> only source encoder <<<")
    eval_tgt(src_encoder, src_classifier, tgt_conv_1_activations_data_loader_eval)

    get_distribution(src_encoder, tgt_encoder, src_classifier, tgt_classifier, critic, src_conv_1_activations_data_loader, 'src')
    get_distribution(src_encoder, tgt_encoder, src_classifier, tgt_classifier, critic, tgt_conv_1_activations_data_loader, 'tgt')

    print(">>> source + target encoders <<<")
    eval_ADDA(src_encoder, tgt_encoder, src_classifier, tgt_classifier, critic, tgt_conv_1_activations_data_loader_eval)

    print(">>> enhanced domain adaptation<<<")
    eval_tgt_with_probe(tgt_encoder, critic, src_classifier, tgt_classifier, tgt_conv_1_activations_data_loader_eval)
'''
