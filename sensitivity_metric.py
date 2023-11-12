import numpy as np
import copy
from tqdm import tqdm
import torch
import torch.nn as nn

import math

import torch.nn.functional as F

from torch.autograd.functional import jacobian
from scipy.linalg import subspace_angles

from scipy.spatial import distance as dis

import scipy

from model import PurchaseClassifier, TexasClassifier, VGG, M18
def multitransp(A):
    """Vectorized matrix transpose.
    ``A`` is assumed to be an array containing ``M`` matrices, each of which
    has dimension ``N x P``.
    That is, ``A`` is an ``M x N x P`` array. Multitransp then returns an array
    containing the ``M`` matrix transposes of the matrices in ``A``, each of
    which will be ``P x N``.
    """
    if A.ndim == 2:
        return A.T
    return np.transpose(A, (0, 2, 1))

def get_sensitivity(function, loader, args, max_batch=None, normalize=False):
    f_norm = 0
    one_norm = 0
    infinite_norm = 0

    k = 0
    for input, labels in tqdm(loader):
        input, labels = input.to('cuda'), labels.to('cuda')
        jac = jacobian(function, input)

        if normalize:
            divider = (torch.max(jac) - torch.min(jac))
        else:
            divider = 1


        f_norm += torch.norm(jac/divider, p='fro')
        one_norm += torch.norm(jac/divider, p=1)
        infinite_norm += torch.norm(jac/divider, p=float("inf"))

        k += 1
        if k == max_batch and max_batch != None:
            break

    f_norm = f_norm / (k*math.sqrt(jac.shape[1]*jac.shape[3]))
    one_norm = one_norm / (k*jac.shape[1]*jac.shape[3])
    infinite_norm = infinite_norm / (k)

    print(f_norm)
    print(one_norm)
    print(infinite_norm)

    return float(f_norm), float(one_norm), float(infinite_norm)
def compute_model_sensitivity_per_layer(model, args):

    sensitivity_per_layer = {}
    if model.isinstance(PurchaseClassifier):
        if len(args.fc_hidden_sizes == 4):
            sensitivity_per_layer['FC1'] = compute_layer_sensitivity(global_model.layer1, t_train_loader, t_test_loader)
            sensitivity_per_layer['FC2'] = compute_layer_sensitivity(global_model.layer2, t_train_loader, t_test_loader)
            sensitivity_per_layer['Classifier'] = compute_layer_sensitivity(global_model.forward, t_train_loader,
                                                                            t_test_loader)

        elif len(args.fc_hidden_sizes == 4):
            sensitivity_per_layer['FC1'] = compute_layer_sensitivity(global_model.layer1, t_train_loader, t_test_loader)
            sensitivity_per_layer['FC2'] = compute_layer_sensitivity(global_model.layer2, t_train_loader, t_test_loader)
            sensitivity_per_layer['FC3'] = compute_layer_sensitivity(global_model.layer3, t_train_loader, t_test_loader)
            sensitivity_per_layer['FC4'] = compute_layer_sensitivity(global_model.layer4, t_train_loader, t_test_loader)
            sensitivity_per_layer['Classifier'] = compute_layer_sensitivity(global_model.forward, t_train_loader, t_test_loader)

    return sensitivity_per_layer


def get_layer_grads(model, layer):
    if layer =='fc1':
        return model.features[0].weight.grad.to('cpu')
    elif layer =='fc2':
        return model.features[2].weight.grad.to('cpu')
    elif layer =='fc3':
        return model.features[4].weight.grad.to('cpu')
    elif layer =='fc4':
        return model.features[6].weight.grad.to('cpu')
    elif layer =='fc5':
        return model.features[8].weight.grad.to('cpu')
    elif layer =='fc6':
        return model.features[10].weight.grad.to('cpu')
    elif layer == 'classifier':
        return model.classifier.weight.grad.to('cpu')

def get_layer_outputs(model, x, layer):
    softmax = torch.nn.Softmax(dim=1)

    if isinstance(model, PurchaseClassifier) or isinstance(model, TexasClassifier):
        if layer =='fc1':
            return softmax(model.layer1(x).to('cpu'))
        elif layer =='fc2':
            return softmax(model.layer2(x).to('cpu'))
        elif layer =='fc3':
            return softmax(model.layer3(x).to('cpu'))
        elif layer =='fc4':
            return softmax(model.layer4(x).to('cpu'))
        elif layer =='fc5':
            return softmax(model.layer5(x).to('cpu'))
        elif layer =='fc6':
            return softmax(model.layer6(x).to('cpu'))
        elif layer == 'classifier':
            return model.forward(x).to('cpu')

    elif isinstance(model, VGG):
        
        if layer == 'conv1':
            return softmax(model.conv1(x).to('cpu'))
        elif layer == 'conv2':
            return softmax(model.conv2(x).to('cpu'))
        elif layer == 'conv3':
            return softmax(model.conv3(x).to('cpu'))
        elif layer == 'conv4':
            return softmax(model.conv4(x).to('cpu'))
        elif layer == 'conv5':
            return softmax(model.conv5(x).to('cpu'))
        elif layer == 'conv6':
            return softmax(model.conv6(x).to('cpu'))
        elif layer == 'conv7':
            return softmax(model.conv7(x).to('cpu'))
        elif layer == 'conv8':
            return softmax(model.conv8(x).to('cpu'))       
        elif layer == 'bn1':
            return softmax(model.bn1(x).to('cpu'))
        elif layer == 'bn2':
            return softmax(model.bn2(x).to('cpu'))
        elif layer == 'bn3':
            return softmax(model.bn3(x).to('cpu'))
        elif layer == 'bn4':
            return softmax(model.bn4(x).to('cpu'))
        elif layer == 'bn5':
            return softmax(model.bn5(x).to('cpu'))
        elif layer == 'bn6':
            return softmax(model.bn6(x).to('cpu'))
        elif layer == 'bn7':
            return softmax(model.bn7(x).to('cpu'))
        elif layer == 'bn8':
            return softmax(model.bn8(x).to('cpu'))

    elif isinstance(model, M18):
        return(softmax(model.hidden_forward(x,layer).to('cpu')))
        




def compute_layer_leakage(model, train_loader, test_loader, args):
    if isinstance(model, VGG):
        return compute_layer_leakage_vgg(model, train_loader, test_loader, args)

    elif isinstance(model, PurchaseClassifier) or isinstance(model, TexasClassifier):
        return compute_layer_leakage_fcnn(model, train_loader, test_loader, args)

def compute_layer_leakage_fcnn(model, train_loader, test_loader, args):
    layer_leakage = {}

    layer_leakage['fc1_f'], layer_leakage['fc1_1'], layer_leakage['fc1_inf'] = get_sensitivity(
        model.layer1, train_loader, args, max_batch=20, normalize=args.normalize_jacobian_norm)

    layer_leakage['fc1_f_test'], layer_leakage['fc1_1_test'], layer_leakage['fc1_inf_test'] = get_sensitivity(
        model.layer1, test_loader,args, max_batch=20, normalize=args.normalize_jacobian_norm)

    layer_leakage['fc2_f'], layer_leakage['fc2_1'], layer_leakage['fc2_inf'] = get_sensitivity(
        model.layer2, train_loader, args, max_batch=20, normalize=args.normalize_jacobian_norm)
    layer_leakage['fc2_f_test'], layer_leakage['fc2_1_test'], layer_leakage['fc2_inf_test'] = get_sensitivity(
        model.layer2, test_loader, args, max_batch=20, normalize=args.normalize_jacobian_norm)

    if len(args.fc_hidden_sizes) >=4:
        layer_leakage['fc3_f'], layer_leakage['fc3_1'], layer_leakage['fc3_inf'] = get_sensitivity(
            model.layer3, train_loader, args, max_batch=20, normalize=args.normalize_jacobian_norm)

        layer_leakage['fc3_f_test'], layer_leakage['fc3_1_test'], layer_leakage['fc3_inf_test'] = get_sensitivity(
            model.layer3, test_loader, args, max_batch=20, normalize=args.normalize_jacobian_norm)

        layer_leakage['fc4_f'], layer_leakage['fc4_1'], layer_leakage['fc4_inf'] = get_sensitivity(
            model.layer4, train_loader, args, max_batch=20, normalize=args.normalize_jacobian_norm)

        layer_leakage['fc4_f_test'], layer_leakage['fc4_1_test'], layer_leakage['fc4_inf_test'] = get_sensitivity(
            model.layer4, test_loader, args, max_batch=20, normalize=args.normalize_jacobian_norm)

    if len(args.fc_hidden_sizes)     >= 6:
        layer_leakage['fc5_f'], layer_leakage['fc5_1'], layer_leakage['fc5_inf'] = get_sensitivity(
            model.layer3, train_loader, args, max_batch=20, normalize=args.normalize_jacobian_norm)

        layer_leakage['fc5_f_test'], layer_leakage['fc5_1_test'], layer_leakage['fc5_inf_test'] = get_sensitivity(
            model.layer3, test_loader, args, max_batch=20, normalize=args.normalize_jacobian_norm)

        layer_leakage['fc6_f'], layer_leakage['fc6_1'], layer_leakage['fc6_inf'] = get_sensitivity(
            model.layer4, train_loader, args,  max_batch=20, normalize=args.normalize_jacobian_norm)

        layer_leakage['fc6_f_test'], layer_leakage['fc6_1_test'], layer_leakage['fc6_inf_test'] = get_sensitivity(
            model.layer4, test_loader, args, max_batch=20, normalize=args.normalize_jacobian_norm)

    layer_leakage['classifier_f'], layer_leakage['classifier_1'], layer_leakage[
        'classifier_inf'] = get_sensitivity(model.forward, train_loader, args, max_batch=20, normalize=args.normalize_jacobian_norm)

    layer_leakage['classifier_f_test'], layer_leakage['classifier_1_test'], layer_leakage[
        'classifier_inf_test'] = get_sensitivity(model.forward, test_loader, args, max_batch=20, normalize=args.normalize_jacobian_norm)

    return layer_leakage


def compute_latent_information_fcnn(model, train_loader, test_loader, args):
    layer_leakage = {}

    layer_leakage['fc1'] = compute_sensitivity_latent_information_outputs(model, train_loader, test_loader, args, 'fc1')
    layer_leakage['fc2'] = compute_sensitivity_latent_information_outputs(model, train_loader, test_loader, args, 'fc2')

    if len(args.fc_hidden_sizes)     >= 4:

        layer_leakage['fc3'] = compute_sensitivity_latent_information_outputs(model, train_loader, test_loader, args, 'fc3')
        layer_leakage['fc4'] = compute_sensitivity_latent_information_outputs(model, train_loader, test_loader, args, 'fc4')

    if len(args.fc_hidden_sizes)     >= 6:
        layer_leakage['fc5'] = compute_sensitivity_latent_information_outputs(model, train_loader, test_loader, args, 'fc5')
        layer_leakage['fc6'] = compute_sensitivity_latent_information_outputs(model, train_loader, test_loader, args, 'fc6')

    layer_leakage['classifier'] = compute_sensitivity_latent_information_outputs(model, train_loader, test_loader,args, 'classifier')

    return layer_leakage

def compute_layer_leakage_vgg(model, train_loader, test_loader, args):

    layer_leakage = {}

    layer_leakage['conv1_f'], layer_leakage['conv1_1'], layer_leakage['conv1_inf'] = get_sensitivity(
        model.conv8, train_loader, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['conv1_f_test'], layer_leakage['conv1_1_test'], layer_leakage['conv1_inf_test'] = get_sensitivity(
        model.conv8, test_loader, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['conv2_f'], layer_leakage['conv2_1'], layer_leakage['conv2_inf'] = get_sensitivity(
        model.conv2, train_loader, max_batch=10, normalize=args.normalize_jacobian_norm)
    layer_leakage['conv2_f_test'], layer_leakage['conv2_1_test'], layer_leakage['conv2_inf_test'] = get_sensitivity(
        model.conv2, test_loader, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['conv3_f'], layer_leakage['conv3_1'], layer_leakage['conv3_inf'] = get_sensitivity(
        model.conv3, train_loader, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['conv3_f_test'], layer_leakage['conv3_1_test'], layer_leakage['conv3_inf_test'] = get_sensitivity(
        model.conv3, test_loader, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['conv4_f'], layer_leakage['conv4_1'], layer_leakage['conv4_inf'] = get_sensitivity(
        model.conv4, train_loader, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['conv4_f_test'], layer_leakage['conv4_1_test'], layer_leakage['conv4_inf_test'] = get_sensitivity(
        model.conv4, test_loader, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['conv5_f'], layer_leakage['conv5_1'], layer_leakage['conv5_inf'] = get_sensitivity(
        model.conv5, train_loader, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['conv5_f_test'], layer_leakage['conv5_1_test'], layer_leakage['conv5_inf_test'] = get_sensitivity(
        model.conv5, test_loader, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['conv6_f'], layer_leakage['conv6_1'], layer_leakage['conv6_inf'] = get_sensitivity(
        model.conv6, train_loader, max_batch=10, normalize=args.normalize_jacobian_norm)
    layer_leakage['conv6_f_test'], layer_leakage['conv6_1_test'], layer_leakage['conv6_inf_test'] = get_sensitivity(
        model.conv6, test_loader, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['conv7_f'], layer_leakage['conv7_1'], layer_leakage['conv7_inf'] = get_sensitivity(
        model.conv7, train_loader, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['conv7_f_test'], layer_leakage['conv7_1_test'], layer_leakage['conv7_inf_test'] = get_sensitivity(
        model.conv7, test_loader, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['conv8_f'], layer_leakage['conv8_1'], layer_leakage['conv8_inf'] = get_sensitivity(
        model.conv8, train_loader, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['conv8_f_test'], layer_leakage['conv8_1_test'], layer_leakage['conv8_inf_test'] = get_sensitivity(
        model.conv8, test_loader, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['bn1_f'], layer_leakage['bn1_1'], layer_leakage['bn1_inf'] = get_sensitivity(
        model.bn8, train_loader, args, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['bn1_f_test'], layer_leakage['bn1_1_test'], layer_leakage['bn1_inf_test'] = get_sensitivity(
        model.bn8, test_loader, args,  max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['bn2_f'], layer_leakage['bn2_1'], layer_leakage['bn2_inf'] = get_sensitivity(
        model.bn2, train_loader, args,  max_batch=10, normalize=args.normalize_jacobian_norm)
    layer_leakage['bn2_f_test'], layer_leakage['bn2_1_test'], layer_leakage['bn2_inf_test'] = get_sensitivity(
        model.bn2, test_loader, args, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['bn3_f'], layer_leakage['bn3_1'], layer_leakage['bn3_inf'] = get_sensitivity(
        model.bn3, train_loader, args, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['bn3_f_test'], layer_leakage['bn3_1_test'], layer_leakage['bn3_inf_test'] = get_sensitivity(
        model.bn3, test_loader, args, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['bn4_f'], layer_leakage['bn4_1'], layer_leakage['bn4_inf'] = get_sensitivity(
        model.bn4, train_loader, args, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['bn4_f_test'], layer_leakage['bn4_1_test'], layer_leakage['bn4_inf_test'] = get_sensitivity(
        model.bn4, test_loader, args, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['bn5_f'], layer_leakage['bn5_1'], layer_leakage['bn5_inf'] = get_sensitivity(
        model.bn5, train_loader, args, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['bn5_f_test'], layer_leakage['bn5_1_test'], layer_leakage['bn5_inf_test'] = get_sensitivity(
        model.bn5, test_loader, args, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['bn6_f'], layer_leakage['bn6_1'], layer_leakage['bn6_inf'] = get_sensitivity(
        model.bn6, train_loader, args, max_batch=10, normalize=args.normalize_jacobian_norm)
    layer_leakage['bn6_f_test'], layer_leakage['bn6_1_test'], layer_leakage['bn6_inf_test'] = get_sensitivity(
        model.bn6, test_loader, args, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['bn7_f'], layer_leakage['bn7_1'], layer_leakage['bn7_inf'] = get_sensitivity(
        model.bn7, train_loader, args, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['bn7_f_test'], layer_leakage['bn7_1_test'], layer_leakage['bn7_inf_test'] = get_sensitivity(
        model.bn7, test_loader, args, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['bn8_f'], layer_leakage['bn8_1'], layer_leakage['bn8_inf'] = get_sensitivity(
        model.bn8, train_loader, args, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['bn8_f_test'], layer_leakage['bn8_1_test'], layer_leakage['bn8_inf_test'] = get_sensitivity(
        model.bn8, test_loader, args, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['classifier_f'], layer_leakage['classifier_1'], layer_leakage[
        'classifier_inf'] = get_sensitivity(model.forward, train_loader, args, max_batch=10, normalize=args.normalize_jacobian_norm)

    layer_leakage['classifier_f_test'], layer_leakage['classifier_1_test'], layer_leakage[
        'classifier_inf_test'] = get_sensitivity(model.forward, test_loader, max_batch=10, normalize=args.normalize_jacobian_norm)

    return layer_leakage


def compute_sensitivity_latent_information_gradients(model, train_loader, test_loader, args,layer, outputs):
    criterion = nn.CrossEntropyLoss().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-4)
    classifier = copy.deepcopy(model)

    with torch.no_grad():
        for input, labels in tqdm(train_loader):

            input, labels = input.to('cuda'), labels.to('cuda')

            output = classifier(input)
            loss = criterion(output.squeeze(), labels)
            loss.backward()
            optimizer.step()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-4)

    classifier = copy.deepcopy(model)
    with torch.no_grad():

        for input, labels in tqdm(test_loader):
            input, labels = input.to('cuda'), labels.to('cuda')

            output = classifier(input)
            loss = criterion(output.squeeze(), labels)
            loss.backward()
            optimizer.step()

    test_grads = get_layer_grads(classifier, layer)


    s = subspace_angles(train_grads, test_grads)
    norm = math.sqrt(sum([x**2 for x in s]))

    return norm

def compute_sensitivity_latent_information_outputs(model, train_loader, test_loader, args, layer,):

    classifier = copy.deepcopy(model)
    train_probs = []
    test_probs = []
    with torch.no_grad():
        for input, labels in tqdm(train_loader):

            input, labels = input.to('cuda'), labels.to('cuda')

            train_outputs = get_layer_outputs(classifier, input, layer)
            train_probs.append(train_outputs)

    classifier = copy.deepcopy(model)
    with torch.no_grad():

        for input, labels in tqdm(test_loader):
            input, labels = input.to('cuda'), labels.to('cuda')


            test_outputs = get_layer_outputs(classifier, input, layer)
            test_probs.append(test_outputs)


    train_grads = torch.cat(train_probs)
    test_grads = torch.cat(test_probs)
    train_grads = train_grads[:test_grads.shape[0]]
    test_grads = test_grads[:train_grads.shape[0]]

    s = subspace_angles(train_grads, test_grads)
    norm = math.sqrt(sum([x**2 for x in s]))

    return norm


def compute_jenssen_shannon_outputs(model, train_loader, test_loader, layer):

    classifier = copy.deepcopy(model)
    train_probs = []
    test_probs = []
    with torch.no_grad():
        limit=0
        for input, labels in tqdm(train_loader):

            limit+=1

            input, labels = input.to('cuda'), labels.to('cuda')

            train_outputs = get_layer_outputs(classifier, input, layer)
            train_probs.append(train_outputs)

            if limit>10:
                break

    classifier = copy.deepcopy(model)
    limit=0
    with torch.no_grad():

        for input, labels in tqdm(test_loader):
            limit+=1

            input, labels = input.to('cuda'), labels.to('cuda')


            test_outputs = get_layer_outputs(classifier, input, layer)
            test_probs.append(test_outputs)

            if limit>10:
                break
    train_grads = torch.cat(train_probs)
    test_grads = torch.cat(test_probs)
    train_grads = train_grads[:test_grads.shape[0]]
    test_grads = test_grads[:train_grads.shape[0]]

    norm = dis.jensenshannon(train_grads, test_grads, axis=0)
    #norm = KL(train_grads, test_grads)
    norm = norm.mean()
    return norm

def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def compute_jenssen_shannon_divergence(model, train_loader, test_loader, args):
    if isinstance(model, PurchaseClassifier) or isinstance(model,TexasClassifier):
        return compute_jenssen_shannon_div_fcnn(model, train_loader, test_loader, args)
    elif isinstance(model, VGG):
        return compute_jenssen_shannon_div_vgg(model, train_loader, test_loader, args)
    elif isinstance(model, M18):
        return compute_jenssen_shannon_div_M18(model, train_loader, test_loader, args)
    else:
        return None

def compute_jenssen_shannon_div_fcnn(model, train_loader, test_loader, args):
    layer_leakage = {}

    layer_leakage['fc1'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'fc1')
    layer_leakage['fc2'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'fc2')

    if len(args.fc_hidden_sizes)     >= 4:

        layer_leakage['fc3'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'fc3')
        layer_leakage['fc4'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'fc4')

    if len(args.fc_hidden_sizes)     >= 6:
        layer_leakage['fc5'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'fc5')
        layer_leakage['fc6'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'fc6')

    layer_leakage['classifier'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'classifier')

    return layer_leakage


def compute_jenssen_shannon_div_vgg(model, train_loader, test_loader, args):
    layer_leakage = {}

    layer_leakage['conv1'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv1')
    layer_leakage['bn1'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn1')
    layer_leakage['conv2'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv2')
    layer_leakage['bn2'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn2')
    layer_leakage['conv3'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv3')
    layer_leakage['bn3'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn3')
    layer_leakage['conv4'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv4')
    layer_leakage['bn4'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn4')
    layer_leakage['conv5'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv5')
    layer_leakage['bn5'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn5')
    layer_leakage['conv6'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv6')
    layer_leakage['bn6'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn6')
    layer_leakage['conv7'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv7')
    layer_leakage['bn7'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn7')
    layer_leakage['conv8'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv8')
    layer_leakage['bn8'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn8')

    return layer_leakage

def compute_jenssen_shannon_div_M18(model, train_loader, test_loader, args):
    layer_leakage = {}

    layer_leakage['conv1'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv1')
    layer_leakage['bn1'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn1')

    layer_leakage['conv2'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv2')
    layer_leakage['bn2'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn2')

    layer_leakage['conv3'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv3')
    layer_leakage['bn3'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn3')

    layer_leakage['conv4'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv4')
    layer_leakage['bn4'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn4')

    layer_leakage['conv5'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv5')
    layer_leakage['bn5'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn5')

    layer_leakage['conv6'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv6')
    layer_leakage['bn6'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn6')

    layer_leakage['conv7'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv7')
    layer_leakage['bn7'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn7')

    layer_leakage['conv8'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv8')
    layer_leakage['bn8'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn8')

    layer_leakage['conv9'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv9')
    layer_leakage['bn9'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn9')

    layer_leakage['conv10'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv10')
    layer_leakage['bn10'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn10')

    layer_leakage['conv11'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv11')
    layer_leakage['bn11'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn11')

    layer_leakage['conv12'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv12')
    layer_leakage['bn12'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn12')

    layer_leakage['conv13'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv13')
    layer_leakage['bn13'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn13')

    layer_leakage['conv14'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv14')
    layer_leakage['bn14'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn14')

    layer_leakage['conv15'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv15')
    layer_leakage['bn15'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn15')

    layer_leakage['conv16'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv16')
    layer_leakage['bn16'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn16')

    layer_leakage['conv17'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'conv17')
    layer_leakage['bn17'] = compute_jenssen_shannon_outputs(model, train_loader, test_loader, 'bn17')

    return layer_leakage
