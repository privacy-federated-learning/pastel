
import numpy as np
import copy
from tqdm import tqdm
import torch
import torch.nn as nn

import math
import scipy
from torch.autograd.functional import jacobian
from scipy.linalg import subspace_angles
from scipy.spatial import distance as dis
from scipy.stats import wasserstein_distance
from model import PurchaseClassifier, TexasClassifier, VGG
from config import PSI, KLD, JSD, EMD

import numpy as np

def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    '''Calculate the PSI (population stability index) across all variables

    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal

    Returns:
       psi_values: ndarray of psi values for each variable

    Author:
       Matthew Burke
       github.com/mwburke
       worksofchart.com
    '''

    def psi(expected_array, actual_array, buckets):
        '''Calculate the PSI for a single variable

        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into

        Returns:
           psi_value: calculated PSI value
        '''

        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input


        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])



        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)

        psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets)

    return(psi_values)


def compute_latent_information_fcnn_jen(model, train_loader, test_loader, args):
    layer_leakage = {}

    layer_leakage['classifier'] = compute_sensitivity_latent_information_gradients_jensenshannon(model, train_loader, test_loader,args, 'classifier')
    layer_leakage['fc1'] = compute_sensitivity_latent_information_gradients_jensenshannon(model, train_loader, test_loader, args, 'fc1')
    layer_leakage['fc2'] = compute_sensitivity_latent_information_gradients_jensenshannon(model, train_loader, test_loader, args, 'fc2')

    if len(args.fc_hidden_sizes) >= 4:
        layer_leakage['fc3'] = compute_sensitivity_latent_information_gradients_jensenshannon(model, train_loader, test_loader, args, 'fc3')
        layer_leakage['fc4'] = compute_sensitivity_latent_information_gradients_jensenshannon(model, train_loader, test_loader, args, 'fc4')

    if len(args.fc_hidden_sizes) >= 6:
        layer_leakage['fc5'] = compute_sensitivity_latent_information_gradients_jensenshannon(model, train_loader, test_loader, args, 'fc5')
        layer_leakage['fc6'] = compute_sensitivity_latent_information_gradients_jensenshannon(model, train_loader, test_loader, args, 'fc6')

    return layer_leakage

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

def compute_sensitivity_latent_information_gradients_jensenshannon(model, train_loader, test_loader, args,layer, distance=EMD):
    criterion = nn.CrossEntropyLoss().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    classifier = copy.deepcopy(model)
    torch.set_grad_enabled(True)
    for input, labels in tqdm(train_loader):
        input, labels = input.to('cuda'), labels.to('cuda')
        output = classifier(input)
        loss = criterion(output.squeeze(), labels)
        loss.backward()
        optimizer.step()
    train_grads = get_layer_grads(classifier, layer)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    classifier = copy.deepcopy(model)

    for input, labels in tqdm(test_loader):
        input, labels = input.to('cuda'), labels.to('cuda')
        output = classifier(input)
        loss = criterion(output.squeeze(), labels)
        loss.backward()
        optimizer.step()
    test_grads = get_layer_grads(classifier, layer)
    if distance == KLD:
        norm = scipy.special.kl_div(train_grads, test_grads)
    elif distance == JSD:
        norm = dis.jensenshannon(train_grads, test_grads, axis=0)
    elif distance == PSI:
        print(train_grads.size())
        print(test_grads.size())
        norm = calculate_psi(train_grads, test_grads)
    elif distance == EMD:
        norm = wasserstein_distance(train_grads, test_grads)

    return norm
