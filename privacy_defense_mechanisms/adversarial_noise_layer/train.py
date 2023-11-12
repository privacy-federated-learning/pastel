import copy
import random

import torch
from torch import nn

import torch.nn.functional as F

from functools import partial
from train import LocalUpdate
from privacy_defense_mechanisms.adversarial_noise_layer.anl import AdvNoise

import sys



class LocalANLUpdate(LocalUpdate):
    def __init__(self, args, dataset, dataset_name, idxs, test_dataset, criterion):
        LocalUpdate.__init__(self, args=args, dataset=dataset, dataset_name=dataset_name, idxs=idxs, test_dataset=test_dataset, criterion=criterion)

    def update_weights(self, model, global_round, model_replacement=False, attack=None):
        # Set mode to train model
        model.train()
        epoch_loss = []
        eps = self.args.eps
        x = copy.deepcopy(model.state_dict())
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        optimizer.zero_grad()
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_loader):

                images, labels = images.to(self.device), labels.to(self.device)

                model.apply(lambda m: type(m) == AdvNoise and m.set_clean())

                if attack:
                    images, labels = self.alter_data_set(images, labels)

                optimizer.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs.squeeze(), labels)
                loss.backward()

                model.apply(lambda m: type(m) == AdvNoise and m.set_stay())
                optimizer.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs.squeeze(), labels)
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

                batch_loss.append(loss.item() * images.size(0))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        if model_replacement:
            return model_replacement(model.state_dict(), x, self.args.num_users, self.args), sum(epoch_loss) / len(
                epoch_loss)

            return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """

        if self.dataset_name in ['cifar', 'cifar100', 'mnist'] :
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