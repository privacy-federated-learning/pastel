import copy
import random

import torch
from torch import nn

import torch.nn.functional as F

from functools import partial
from train import LocalUpdate

import numpy as np
import copy
import os

import torch
import torch.nn.functional as Func
from dataset import DatasetSplit, filter_celeba_by_indices
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from utils import model_replacement, attack_test_visual_pattern
from data.motionsense.load_data import motionsense_collate_fn
from data.speech_commands.load_data import speech_commands_collate_fn
from data.celeba.load_data import celeba_collate_fn
from data.purchase.purchase import purchase_collate_fn
import random

import sys

from privacy_defense_mechanisms.relaxloss.source.utils import mkdir, str2bool, write_yaml, load_yaml, adjust_learning_rate, \
    AverageMeter, Bar, plot_hist, accuracy, one_hot_embedding, CrossEntropy_soft, NLL_soft


class LocalRelaxLossUpdate(LocalUpdate):
    def __init__(self, args, dataset, dataset_name, idxs, logger, test_dataset, alpha, num_classes, upper ):
        LocalUpdate.__init__(self, args=args, dataset=dataset, dataset_name=dataset_name, idxs=idxs, test_dataset=test_dataset)
        self.alpha = alpha
        self.num_classes = num_classes
        self.upper = upper

        self.device = 'cuda' if self.args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.train_loader, self.valid_loader, self.test_loader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.test_dataset = test_dataset



        if self.args.dataset == 'speech_commands':
            self.criterion = nn.NLLLoss().to(self.device)
            self.criterion_noreduce = nn.NLLLoss(reduction = 'none').to(self.device)
            self.criterion_soft = partial(NLL_soft, reduction='none')
        else:
            self.criterion = nn.CrossEntropyLoss().to(self.device)
            self.criterion_noreduce = nn.CrossEntropyLoss(reduction='none').to(self.device)
            self.criterion_soft = partial(CrossEntropy_soft, reduction='none')


    def update_weights(self, model, global_round, model_replacement=False, attack=None):
        # Set mode to train model
        model.train()
        epoch_loss = []
        accuracies = []
        eps = self.args.eps
        x = copy.deepcopy(model.state_dict())
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        elif self.args.optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        losses = AverageMeter()
        losses_ce = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        label_list = []
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            label_list.append(labels)

        label_list = torch.cat(label_list)
        uniques = label_list.unique(return_counts=True)
        distribution_per_class  = torch.zeros(self.args.num_classes)

        for i in range(uniques[0].shape[0]):
            distribution_per_class[int(uniques[0][i])] = uniques[1][i]

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()  # print(images.shape)

                if self.args.dataset == 'speech_commands':
                    log_probs = model(images).squeeze()
                    loss_full = self.criterion(log_probs.squeeze(), labels)
                else:
                    log_probs = model(images.squeeze()).squeeze()
                    loss_full = self.criterion(log_probs.squeeze(), labels)

                loss_ce = loss_full

                if loss_ce > self.alpha:  # normal gradient descent
                    loss = loss_ce
                else:
                    if iter % 2 == 0:  # gradient ascent/ normal gradient descent
                        loss = (loss_ce - self.alpha).abs()
                    else:  # posterior flattening
                        pred = torch.argmax(log_probs.squeeze(), dim=1)
                        correct = torch.eq(pred, labels).float()
                        confidence_target = F.softmax(log_probs, dim=1)[torch.arange(labels.size(0)), labels]
                        confidence_target = torch.clamp(confidence_target, min=0., max=self.upper)

                        onehot = one_hot_embedding(labels, num_classes=self.num_classes)

                        if self.args.prob_distribution == 'normal':
                            mean = (distribution_per_class*1.0).mean()
                            std = (distribution_per_class*1.0).std()
                            soft_probs = torch.Tensor(np.random.normal(mean, std, self.num_classes)).to('cuda')
                            confidence_else = F.softmax(soft_probs.repeat(images.shape[0], 1), dim=1)
                            soft_targets = onehot * confidence_target.unsqueeze(-1).repeat(1, self.num_classes) \
                                           + (1 - onehot) * confidence_else

                        elif self.args.prob_distribution == 'weighted':
                            confidence_else = ((1.0 - confidence_target) / (self.num_classes - 1)).repeat(1, self.num_classes)
                            coeffs =  (distribution_per_class/distribution_per_class.sum())
                            confidence_else = (confidence_else.repeat(32,1).T).matmul(coeffs)
                            soft_targets = onehot * confidence_target.unsqueeze(-1).repeat(1, self.num_classes) \
                                           + (1 - onehot) * confidence_else

                        else:
                            confidence_else = (1.0 - confidence_target) / (self.num_classes - 1)

                            soft_targets = onehot * confidence_target.unsqueeze(-1).repeat(1, self.num_classes) \
                                           + (1 - onehot) * confidence_else.unsqueeze(-1).repeat(1, self.num_classes)
                        loss = self.criterion_soft(log_probs.squeeze(), soft_targets)
                        loss = torch.mean(loss)

                ### Record accuracy and loss
                #prec1, prec5 = accuracy(log_probs.data, labels.data, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                losses_ce.update(loss_ce.item(), images.size(0))
                #top1.update(prec1.item(), images.size(0))
                #top5.update(prec5.item(), images.size(0))

                loss.backward()
                optimizer.step()

                # @Rania : Added this condition
                if attack and self.args.pgd:
                    x_adv = copy.deepcopy(model.state_dict())
                    for key in x_adv.keys():
                        x_adv[key] = torch.max(torch.min(x_adv[key], x[key] + eps), x[key] - eps)
                    model.load_state_dict(x_adv)

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.train_loader.dataset),
                                            100. * batch_idx / len(self.train_loader), loss.item()))
                # self.logger.add_scalar('loss', loss.item())

                batch_loss.append(loss.item())
            accur, _ = self.test_inference(copy.deepcopy(model), self.test_loader)
            #print("Accuracy : ",accuracy)
            accuracies.append(accur)

            epoch_loss.append(sum(batch_loss) / len(batch_loss))


        if model_replacement:
            return model_replacement(model.state_dict(), x, self.args.num_users, self.args), sum(epoch_loss) / len(
                epoch_loss)
        else:
            return model.state_dict(), sum(epoch_loss) / len(epoch_loss), accuracies

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """

        if self.dataset_name in ['cifar', 'cifar100', 'mnist']:
            train_loader, valid_loader, test_loader = self.train_val_test_common(dataset, idxs)

        elif self.dataset_name in ['purchase', 'texas', 'gtsrb']:
            train_loader, valid_loader, test_loader = self.train_val_test_purchase(dataset, idxs)

        elif self.dataset_name == 'motionsense':
            train_loader, valid_loader, test_loader = self.train_val_test_motionsense(dataset, idxs)

        elif self.dataset_name == 'speech_commands':
            train_loader, valid_loader, test_loader = self.train_val_test_speech_commands(dataset, idxs)

        elif self.dataset_name == 'celeba':
            train_loader, valid_loader, test_loader = self.train_val_test_celeba(dataset, idxs)

        return train_loader, valid_loader, test_loader

    def train_val_test_speech_commands(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """

        random.Random(4).shuffle(idxs)
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)):]

        batch_size = 1024

        num_workers = 2
        pin_memory = True

        trainloader = torch.utils.data.DataLoader(
            DatasetSplit(dataset, idxs_train),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=speech_commands_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        testloader = torch.utils.data.DataLoader(
            DatasetSplit(dataset, idxs_test),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=speech_commands_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        validloader = torch.utils.data.DataLoader(
            DatasetSplit(dataset, idxs_val),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=speech_commands_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return trainloader, validloader, testloader
