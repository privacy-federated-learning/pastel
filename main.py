import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

import config
from aggregation import FLaggregate, average_weights
from config import PASTEL, GAUSSIAN_NOISE
from dataset import get_dataset, split_target_shadow_dataset, data_to_loader
from mia import create_attack
from model import get_models
from parser import Arguments
from pastel import pastel, aggregation_pastel
from train import LocalUpdate
from utils import test_inference

from privacy_defense_mechanisms.relaxloss.train import LocalRelaxLossUpdate
from privacy_defense_mechanisms.adversarial_noise_layer.train import LocalANLUpdate


import sys

import time


def display_users(idx, epoch):
    print('------------------------------------------')
    print(f'------User: {idx}, Epoch: {epoch}--------')
    print('------------------------------------------')


def select_attack():
    return False


def aggregate_models(local_weights, args):
    """
    :param local_weights: client updates
    :param args: system settup
    :return: aggregated model
    """
    if args.ppm == PASTEL or PASTEL in args.ppm:
        print("PASTEL OK")
        return aggregation_pastel(local_weights, args.pastel_layers)
    return average_weights(local_weights)


def save_model(args, model, epoch):
    if args.save_model:
        torch.save(model, f'{args.path}_{epoch}.pt')


def display_setup(args):
    print(f'--- Launching Exp for {args.dataset}--------')


def get_criterion(args):

    if args.criterion == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif args.criterion == 'nll':
        return F.nll_loss
    elif args.dataset in ['motionsense', 'purchase', 'texas']:
        return nn.CrossEntropyLoss()
    else:
        return F.nll_loss


def fl_training(args):
    train_loss, train_accuracy, test_accuracy, list_acc, list_loss = [], [], [], [], []
    train_dataset, test_dataset, user_groups = get_dataset(args)
    global_model = get_models(args)
    criterion = get_criterion(args)
    global_model.to(device)
    global_model.train()
    global_weights = global_model.state_dict()
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        # select users
        m = max(int(args.frac * args.num_users), 1)
        idx_users = np.random.choice(range(args.num_users), m, replace=False)
        print(f'Selected users for the training: {idx_users}')
        for idx in idx_users:
            display_users(idx, epoch)
            client = LocalUpdate(args=args, dataset=t_train_dataset, idxs=user_groups[idx], test_dataset=test_dataset,
                                 criterion=criterion)
            w, loss = client.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, attack=select_attack())
            acc, loss = client.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
        global_weights = aggregate_models(local_weights, args)
        save_model(args, global_weights, epoch)
        global_model.load_state_dict(global_weights)
        train_accuracy.append(sum(list_acc) / len(list_acc))
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        test_accuracy.append(test_acc)
        print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss))}')
        print('Train Accuracy: {:.2f}% \n'.format(train_accuracy[-1]))

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_accuracy[-1]))


def fl_training_ppm(args, device):
    train_loss, train_accuracy, test_accuracy, list_acc, list_loss = [], [], [], [], []

    train_dataset, test_dataset, user_groups = get_dataset(args)

    t_train_dataset, t_test_dataset, t_val_dataset, s_train_dataset, s_test_dataset, s_val_dataset, user_groups = split_target_shadow_dataset(train_dataset, args)

    t_train_loader = data_to_loader(t_train_dataset, args)
    t_test_loader = data_to_loader(t_test_dataset, args)
    t_val_loader = data_to_loader(t_val_dataset, args)

    s_train_loader = data_to_loader(s_train_dataset, args)
    s_test_loader = data_to_loader(s_test_dataset, args)
    s_val_loader = data_to_loader(s_val_dataset, args)


    global_model = get_models(args)
    client_model = get_models(args)
    criterion = get_criterion(args)
    client_model.to(device)
    global_model.to(device)
    global_model.train()
    clients_accuracy = defaultdict(list)
    server_accuracy = []
    clients_attack = defaultdict(list)
    server_attack = []
    clients_auc = defaultdict(list)
    server_auc = []
    client_models = [global_model] * 10
    client_accuracies = []
    for epoch in tqdm(range(args.epochs)):

        # if epoch >= 20:
        #     args.lr = 1e-4
        round_start = time.time()

        local_weights, local_losses = [], []
        # select users
        m = max(int(args.frac * args.num_users), 1)
        idx_users = np.random.choice(range(args.num_users), m, replace=False)
        print(f'Selected users for the training: {idx_users}')
        for idx in idx_users:
            display_users(idx, epoch)

            if args.ppm == 'relaxloss' or 'relaxloss' in args.ppm:
                print("Relaxloss OK")
                client = LocalRelaxLossUpdate(args=args, dataset=train_dataset, dataset_name=args.dataset, idxs=user_groups[idx],
                                                   test_dataset=test_dataset, logger='', alpha=args.relaxloss_alpha,
                                                   num_classes = args.num_classes, upper=1)

            elif args.ppm == 'anl':
                client = LocalANLUpdate(args=args, dataset=t_train_dataset, dataset_name=args.dataset,
                                     idxs=user_groups[idx],
                                     test_dataset=test_dataset, criterion=criterion)
            else:

                client = LocalUpdate(args=args, dataset=t_train_dataset, dataset_name=args.dataset, idxs=user_groups[idx],
                                     test_dataset=test_dataset,criterion=criterion)

            w, loss = client.update_weights(
                model=copy.deepcopy(client_models[idx]), global_round=epoch, attack=select_attack())
            client_model.load_state_dict(w)
            acc, loss = client.test_inference(model=client_model, loader=t_test_loader)
            list_acc.append(acc)
            clients_accuracy[f"client_accuracy_{idx}"].append(acc)
            print(clients_accuracy)
            list_loss.append(loss)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))


            if epoch >= args.attack_min_epoch:
                attack_accuracy, auc = ppm_eval(client.train_loader, client.test_loader, client.valid_loader,
                                                                       s_train_loader, s_test_loader, s_val_loader,
                                                                       args, w)

                clients_attack[f"client_attack_{idx}"].append(attack_accuracy)
                clients_auc[f"client_auc_{idx}"].append(auc)

            else:
                clients_attack[f"client_attack_{idx}"].append(0)
                clients_auc[f"client_auc_{idx}"].append(0)

            local_losses.append(copy.deepcopy(loss))

        global_weights, global_weights_list = aggregate_models(local_weights, args)
        save_model(args, global_weights, epoch)
        global_model.load_state_dict(global_weights)
        if global_weights_list:
            for idx, client_weight in enumerate(global_weights_list):
                client_models[idx].load_state_dict(client_weight)
        #         client_accuracies.append(test_inference(args, client_models[idx], test_dataset))
        print(client_accuracies)
        train_accuracy.append(sum(list_acc) / len(list_acc))
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        test_accuracy.append(test_acc)
        server_accuracy.append(test_acc)
        print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss))}')

        print('Train Accuracy: {:.2f}% \n'.format(train_accuracy[-1]))

        if epoch >= args.attack_min_epoch:
            server_mia, auc = ppm_eval(t_train_loader, t_test_loader, t_val_loader,
                                  s_train_loader, s_test_loader, s_val_loader,
                                  args, global_weights)
            server_attack.append(server_mia)
            server_auc.append(auc)
        else:
            server_attack.append(0)
            server_auc.append(0)

        round_end = time.time()

        print("Round executed in {}".format(str(round_end - round_start)))

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_accuracy[-1]))
    print(server_attack)
    print(server_accuracy)
    print(clients_accuracy)
    print(clients_attack)
    save_data(args, clients_accuracy, server_accuracy, clients_attack, server_attack, clients_auc, server_auc)


def ppm_eval(t_train_loader, t_test_loader, t_val_loader, s_train_loader, s_test_loader, s_val_loader, args, model):
    if args.ppm in config.PASTEL_CONFIG or 'pastel' in args.ppm:
        print("PASTEL STILL OK")
        model = pastel(model, args.pastel_layers)
    return create_attack(t_train_loader, t_test_loader, t_val_loader, s_train_loader, s_test_loader, s_val_loader, args, model)


def save_data(args, clients_accuracy, server_accuracy, clients_attack, server_attack, clients_auc, server_auc):
    df_tmp = pd.DataFrame.from_dict(clients_accuracy)
    df_tmp_1 = pd.DataFrame.from_dict(clients_attack)
    df_tmp_2 = pd.DataFrame.from_dict(clients_auc)
    df_tmp_3 = pd.DataFrame.from_dict({"server_attack": server_attack})
    df_tmp_4 = pd.DataFrame.from_dict({"server_accuracy": server_accuracy})
    df_tmp_5 = pd.DataFrame.from_dict({"server_auc": server_auc})
    df = pd.concat([df_tmp, df_tmp_1, df_tmp_2, df_tmp_3, df_tmp_4, df_tmp_5], axis=1)
    df.to_csv(args.result_path)


if __name__ == '__main__':
    args = Arguments()
    device = 'cuda' if args.gpu else 'cpu'
    # args.epochs = 50
    # args.ppm = 'gnl'

    exp_start = time.time()

    fl_training_ppm(args, device)

    exp_end = time.time()
    print("Experiment executed in {}".format(str(exp_end - exp_start)))
    # fl_training(args)





"""
TO DO
Chnage names of clients in the dataframe

"""