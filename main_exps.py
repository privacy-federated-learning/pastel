import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

import config
from aggregation import  average_weights
from config import PASTEL
from dataset import get_dataset, split_target_shadow_dataset, data_to_loader
from mia import create_attack, prepare_attack_model
from model import get_models
from parser import Arguments
from pastel import pastel, aggregation_pastel
from train import LocalUpdate
from train_ldp import Opacus_LocalUpdate
from train_wdp import WDPLocalUpdate

from sensitivity_metric import compute_jenssen_shannon_divergence

from utils import test_inference
from cdp import update_cdp


import sys

sys.path.append('model_transfer_exp')


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
    local_weights = copy.deepcopy(local_weights)
    if args.ppm == PASTEL or 'pastel' in args.ppm:
        return aggregation_pastel(local_weights, args.pastel_layers)
    return average_weights(local_weights, args)


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
    train_dataset, test_dataset, user_groups, test_user_groups = get_dataset(args)
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
    clients_loss = defaultdict(list)


    clients_layer_leakage = defaultdict(list)
    server_layer_leakage = []
    server_accuracy = []

    clients_accuracy_per_epoch = defaultdict(list)
    clients_loss_per_epoch = defaultdict(list)
    server_accuracy = []
    clients_attack = defaultdict(list)
    server_attack = []
    clients_auc = defaultdict(list)
    server_auc = []
    clients_round_duration = defaultdict(list)
    server_aggregation_duration = []

    clients_training_memory_usage = defaultdict(list)

    clients_model_parameters_memory_usage = defaultdict(list)
    server_model_parameters_memory_usage = []

    clients_data_parameters_memory_usage = defaultdict(list)
    server_data_parameters_memory_usage = []

    clients_transformation_time = defaultdict(list)
    server_transformation_time = []

    client_models = [global_model] * 10
    client_accuracies = []

    hidden_sizes = [32, 64, 128]
    batch_size = 64
    lr = 1e-3
    lmbda = 0.6
    weight_decay = 5e-4
    nb_epochs = 20
    depth = 7
    feature_size = args.feature_size


    for epoch in tqdm(range(args.epochs)):



        for i in range(len(client_models)):
            clients_transformation_time[f"client_transformation_time_{i}"].append(0)
        server_transformation_time.append(0)


        #
        # if epoch >= 40:
        #     args.lr = 1e-4
        round_start = time.time()

        local_weights, local_losses = [], []
        # select users
        m = max(int(args.frac * args.num_users), 1)
        idx_users = np.random.choice(range(args.num_users), m, replace=False)
        print(f'Selected users for the training: {idx_users}')
        #
        if (epoch == args.attack_min_epoch):
            prepare_attack_model(s_train_loader, s_test_loader, s_val_loader, args)


        for idx in idx_users:

            client_round_start = time.time()


            if args.ppm == 'ldp':

                client = Opacus_LocalUpdate(args=args, dataset=t_train_dataset, dataset_name=args.dataset, idxs=user_groups[idx],
                                     test_dataset=test_dataset,criterion=criterion)

            elif args.ppm == 'wdp':
                client = WDPLocalUpdate(args=args, dataset=t_train_dataset, dataset_name=args.dataset, idxs=user_groups[idx],
                                     test_dataset=test_dataset,criterion=criterion)

            else:

                client = LocalUpdate(args=args, dataset=t_train_dataset, dataset_name=args.dataset, idxs=user_groups[idx],
                                     test_dataset=test_dataset, criterion=criterion)


            w, training_loss, accuracies_per_epoch, losses_per_epoch = client.update_weights(
                model=copy.deepcopy(client_models[idx]), global_round=epoch, attack=select_attack())
            client_model.load_state_dict(w)

            if args.ppm == 'pastel_dp':
                obfuscated_weights = client.obfuscate_weights(w, model=copy.deepcopy(client_models[idx]), global_round=epoch, attack=select_attack())
            else:
                obfuscated_weights = None

            acc, loss = client.test_inference(model=client_model, loader=t_test_loader)
            accuracies_per_epoch[-1] = acc
            client_round_end = time.time()

            list_acc.append(acc)
            clients_accuracy_per_epoch[f"client_accuracy_per_epoch{idx}"].append(accuracies_per_epoch)
            clients_loss_per_epoch[f"client_loss_per_epoch{idx}"].append(losses_per_epoch)

            clients_accuracy[f"client_accuracy_{idx}"].append(acc)

            clients_loss[f"client_loss_{idx}"].append(training_loss)

            print(clients_accuracy)

            clients_round_duration[f"client_round_duration_{idx}"].append(client_round_end - client_round_start)

            list_loss.append(training_loss)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

            if epoch == args.epochs -1 and args.measure_layer_latent_information:
                new_args = copy.deepcopy(args)
                new_args.local_bs = args.jacobian_bs

                jacobian_client = LocalUpdate(args=new_args, dataset=t_train_dataset, dataset_name=args.dataset, idxs=user_groups[idx],
                                     test_dataset=test_dataset,criterion=criterion)

                #layer_leakage = compute_layer_leakage(client_model, jacobian_client.train_loader, jacobian_client.test_loader, args)
                #layer_leakage = compute_latent_information_fcnn(client_model, client.train_loader, client.test_loader, args)
                layer_leakage = compute_jenssen_shannon_divergence(client_model, client.train_loader, client.test_loader, args)
                clients_layer_leakage[f"client_layer_leakage_{idx}"].append(layer_leakage)

            else:
                clients_layer_leakage[f"client_layer_leakage_{idx}"].append({})

            if epoch >= args.attack_min_epoch:
                attack_accuracy, auc = ppm_eval(client.train_loader, client.test_loader, client.valid_loader,
                                                                       s_train_loader, s_test_loader, s_val_loader,
                                                                       args, w, obfuscated_weights)

                clients_attack[f"client_attack_{idx}"].append(attack_accuracy)
                clients_auc[f"client_auc_{idx}"].append(auc)

            else:
                clients_attack[f"client_attack_{idx}"].append(0)
                clients_auc[f"client_auc_{idx}"].append(0)

            local_losses.append(copy.deepcopy(loss))



        server_aggregation_start = time.time()

        global_weights, global_weights_list = aggregate_models(local_weights, args)
        save_model(args, global_weights, epoch)


        global_model.load_state_dict(global_weights)

        if args.ppm == 'cdp':
            global_model = update_cdp(global_model, t_train_loader.dataset, args)

        server_aggregation_end = time.time()

        server_aggregation_duration.append(server_aggregation_end - server_aggregation_start)

        if global_weights_list:
            for idx, client_weight in enumerate(global_weights_list):
                client = copy.deepcopy(client_models[idx])
                client.load_state_dict(client_weight)
                client_models[idx] = client
        #         client_accuracies.append(test_inference(args, client_models[idx], test_dataset))
        print(client_accuracies)
        train_accuracy.append(sum(list_acc) / len(list_acc))


        test_acc, test_loss = test_inference(args, global_model, test_dataset)

        test_accuracy.append(test_acc)
        server_accuracy.append(test_acc)
        print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss))}')

        print('Train Accuracy: {:.2f}% \n'.format(train_accuracy[-1]))

        if epoch == args.epochs - 1 and args.measure_layer_latent_information:

            #layer_leakage = compute_layer_leakage(global_model, jacobian_train_loader, jacobian_test_loader, args)
            #layer_leakage = compute_latent_information_fcnn(global_model, t_train_loader, t_test_loader, args)
            layer_leakage = compute_jenssen_shannon_divergence(global_model, t_train_loader, t_test_loader, args)

            server_layer_leakage.append(layer_leakage)
        else:
            server_layer_leakage.append({})

        if epoch >= args.attack_min_epoch:
            server_mia, auc = ppm_eval(t_train_loader, t_test_loader, t_val_loader,
                                  s_train_loader, s_test_loader, s_val_loader,
                                  args, global_weights, obfuscated_weights)
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
    save_data(args, clients_accuracy, clients_loss, clients_accuracy_per_epoch, clients_loss_per_epoch, server_accuracy, clients_attack, server_attack, clients_auc, server_auc, clients_layer_leakage, server_layer_leakage,
              clients_round_duration, server_aggregation_duration, server_model_parameters_memory_usage, clients_model_parameters_memory_usage,
              server_data_parameters_memory_usage, clients_data_parameters_memory_usage, clients_training_memory_usage,
              server_transformation_time, clients_transformation_time
              )



def ppm_eval(t_train_loader, t_test_loader, t_val_loader, s_train_loader, s_test_loader, s_val_loader, args, model, obfuscated_weights):
    if args.ppm in config.PASTEL_CONFIG or 'pastel' in args.ppm:
        model = pastel(model, args.layer_type, args.pastel_layers)
    if 'pastel_dp' in args.ppm:
        model = obfuscated_weights

    return create_attack(t_train_loader, t_test_loader, t_val_loader, s_train_loader, s_test_loader, s_val_loader, args, model)


def save_data(args, clients_accuracy, clients_loss, clients_accuracy_per_epoch, clients_loss_per_epoch, server_accuracy, clients_attack, server_attack, clients_auc, server_auc,
              clients_layer_leakage, server_layer_leakage,
              clients_round_duration, server_aggregation_duration, server_model_parameters_memory_usage, clients_model_parameters_memory_usage,
              server_data_parameters_memory_usage, clients_data_parameters_memory_usage, clients_training_memory_usage,
              server_transformation_time, clients_transformation_time):
    df_tmp = [pd.DataFrame.from_dict(clients_accuracy)]
    df_tmp.append(pd.DataFrame.from_dict(clients_loss))

    df_tmp.append(pd.DataFrame.from_dict(clients_accuracy_per_epoch))
    df_tmp.append(pd.DataFrame.from_dict(clients_loss_per_epoch))
    df_tmp.append(pd.DataFrame.from_dict(clients_attack))
    df_tmp.append(pd.DataFrame.from_dict(clients_auc))
    df_tmp.append(pd.DataFrame.from_dict({"server_attack": server_attack}))
    df_tmp.append(pd.DataFrame.from_dict({"server_accuracy": server_accuracy}))
    df_tmp.append(pd.DataFrame.from_dict({"server_auc": server_auc}))
    df_tmp.append(pd.DataFrame.from_dict(clients_layer_leakage))

    df_tmp.append(pd.DataFrame.from_dict({"server_layer_leakage": server_layer_leakage}))
    df_tmp.append(pd.DataFrame.from_dict(clients_round_duration))
    df_tmp.append(pd.DataFrame.from_dict(clients_data_parameters_memory_usage))
    df_tmp.append(pd.DataFrame.from_dict(clients_model_parameters_memory_usage))
    df_tmp.append(pd.DataFrame.from_dict(clients_training_memory_usage))
    df_tmp.append(pd.DataFrame.from_dict(clients_transformation_time))
    df_tmp.append(pd.DataFrame.from_dict({"server_aggregation_duration": server_aggregation_duration}))
    df_tmp.append(pd.DataFrame.from_dict({"server_model_parameters_memory_usage": server_model_parameters_memory_usage}))
    df_tmp.append(pd.DataFrame.from_dict({"server_data_parameters_memory_usage": server_data_parameters_memory_usage}))
    df_tmp.append(pd.DataFrame.from_dict({"server_transformation_time": server_transformation_time}))
    df = pd.concat(df_tmp, axis=1)
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
