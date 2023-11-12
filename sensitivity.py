

import torch
from torch import nn
import types
from functools import partial
import torch.nn.functional as F
import argparse
from tqdm import tqdm

from torch.autograd.functional import jacobian
from torchvision import transforms
from torchvision import datasets
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from models.resnet import resnet18
import dataset_mia
import pandas as pd

def get_sensitivity(function, train_loader, train='member'):
    f_norm=0
    one_norm = 0
    f_norms = []
    one_norms = []
    k=0
    for input,_ in tqdm(train_loader):
        ## calculate jacobian
        input = input.to('cuda')
        jac = jacobian(function, input)
        ## calculate norm
        tmp_f_norm = torch.norm(jac, p='fro')
        tmp_one_norm = torch.norm(jac, p=1)
        ## add norms
        f_norm += tmp_f_norm
        one_norm += tmp_one_norm
        print(f_norm)
        print(one_norm)
        ## append norm
        f_norms.append(tmp_f_norm.cpu())
        one_norms.append(tmp_one_norm.cpu())
        k+=1
        if k == 5000:
            break
    dict_ = {'fnorm': f_norms, 'one_norm': one_norms}
    save_sensitivity(dict_, f'results/{function.__name__}_{train}.csv')
    f_norm = f_norm / k
    one_norm = one_norm / k



    return f_norm, one_norm

def save_sensitivity(dict, path):
    df_tmp = pd.DataFrame.from_dict(dict)
    df_tmp.to_csv(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training args
    parser.add_argument('-net', type=str, help='net type')
    parser.add_argument('--epochs', type=int, default=400, help="number of rounds of training")
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')

    # federated learning args
    parser.add_argument('--iid', type=int, default=1, help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--num_users', type=int, default=10, help='Number of clients')
    parser.add_argument('--frac', type=float, default=1.0, help='Fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=1, help='Local training epochs for each client')

    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--local_bs', type=int, default=1, help='atch size')
    parser.add_argument('--attack_min_epoch', type=int, default=100, help='atch size')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--ppm', type=str, default='pastel')
    parser.add_argument('--pastel_layers', type=str, default='bn')
    parser.add_argument('--data_path', type=str, default='./attack_model')
    parser.add_argument('--attack_directory', type=str, default='./attack_model')
    parser.add_argument('--model_path', type=str, default='.')
    parser.add_argument('--need_topk', action='store_true', default=False)
    parser.add_argument('--train_attack_model',  action='store_true', default=True)
    parser.add_argument('--train_shadow_model', action='store_true', default=True)
    parser.add_argument('--param_init', action='store_true', default=True)
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--criterion', type=str, default='cross_entropy')
    args = parser.parse_args()

    net = resnet18()
    net.load_state_dict(torch.load('fed_train_100.pt'))
    net.to('cuda')
    train_dataset, test_dataset, user_groups = dataset_mia.get_dataset(args)
    train_loader = dataset_mia.data_to_loader(train_dataset, args)
    test_loader = dataset_mia.data_to_loader(test_dataset, args)
    print(get_sensitivity(net.forward_bn1, test_loader, 'non_member'))
    print(get_sensitivity(net.forward_bn1, train_loader, 'member'))
    print(get_sensitivity(net.forward_conv1, test_loader, 'non_member'))
    print(get_sensitivity(net.forward_conv1, train_loader, 'member'))