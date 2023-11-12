import copy
import os

import torch
import torch.nn.functional as Func

from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise

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

def gaussian_noise(data_shape, s, sigma, device=None):
    """
    Gaussian noise
    """
    return torch.normal(0, sigma * s, data_shape).to(device)

class DP_LocalUpdate(object):
    def __init__(self, args, dataset, dataset_name, idxs, test_dataset, criterion=nn.NLLLoss()):
        self.args = args

        self.dataset_name = dataset_name

        self.train_loader, self.valid_loader, self.test_loader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = criterion
        self.test_dataset = test_dataset


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


    def train_val_test_common(self, dataset, idxs):
        # split indexes for train, validation, and test (80, 10, 10)
        idx_train = idxs[:int(0.8 * len(idxs))]
        idx_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
        idx_test = idxs[int(0.9 * len(idxs)):]
        train_loader = DataLoader(DatasetSplit(dataset, idx_train),
                                  batch_size=self.args.local_bs, shuffle=True)
        valid_loader = DataLoader(DatasetSplit(dataset, idx_val),
                                  batch_size=int(len(idx_val) / 10), shuffle=False)
        test_loader = DataLoader(DatasetSplit(dataset, idx_test),
                                 batch_size=int(len(idx_test) / 10), shuffle=False)

        return train_loader, valid_loader, test_loader


    def train_val_test_celeba(self, dataset, idxs):
        # split indexes for train, validation, and test (80, 10, 10)
        random.shuffle(idxs)
        idx_train = idxs[:int(0.8 * len(idxs))]
        idx_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
        idx_test = idxs[int(0.9 * len(idxs)):]

        train_loader = DataLoader(filter_celeba_by_indices(dataset, idx_train),
                                  batch_size=self.args.local_bs,
                                  collate_fn = celeba_collate_fn,
                                  shuffle=True)
        valid_loader = DataLoader(filter_celeba_by_indices(dataset, idx_val),
                                  batch_size=int(len(idx_val) / 10),
                                  collate_fn=celeba_collate_fn,
                                  shuffle=False)
        test_loader = DataLoader(filter_celeba_by_indices(dataset, idx_test),
                                 batch_size=int(len(idx_test) / 10),
                                 collate_fn=celeba_collate_fn,
                                 shuffle=False)

        return train_loader, valid_loader, test_loader


    def train_val_test_purchase(self, dataset, idxs):
        # split indexes for train, validation, and test (80, 10, 10)
        idx_train = idxs[:int(0.8 * len(idxs))]
        idx_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
        idx_test = idxs[int(0.9 * len(idxs)):]

        train_loader = DataLoader(DatasetSplit(dataset, idx_train),
                                  collate_fn = None,
                                  batch_size=self.args.local_bs, shuffle=True)
        valid_loader = DataLoader(DatasetSplit(dataset, idx_val),
                                  collate_fn = None,
                                  batch_size=self.args.local_bs, shuffle=False)
        test_loader = DataLoader(DatasetSplit(dataset, idx_test),
                                  collate_fn = None,
                                 batch_size=self.args.local_bs, shuffle=False)

        return train_loader, valid_loader, test_loader

    def train_val_test_motionsense(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idx_train = idxs[:int(0.8 * len(idxs))]
        idx_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
        idx_test = idxs[int(0.9 * len(idxs)):]

        train_dataset = copy.deepcopy(dataset)
        train_dataset._walker = [train_dataset._walker[i] for i in idx_train]

        test_dataset = copy.deepcopy(dataset)
        test_dataset._walker = [test_dataset._walker[i] for i in idx_test]

        val_dataset = copy.deepcopy(dataset)
        val_dataset._walker = [val_dataset._walker[i] for i in idx_val]

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.args.local_bs,
                                  collate_fn=motionsense_collate_fn,
                                  shuffle=True)
        valid_loader = DataLoader(val_dataset,
                                  batch_size=int(len(idx_val) / 10),
                                  collate_fn=motionsense_collate_fn,
                                  shuffle=False)
        test_loader = DataLoader(test_dataset,
                                 batch_size=int(len(idx_test) / 10),
                                 collate_fn=motionsense_collate_fn,
                                 shuffle=False)

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

    def add_visual_pattern(self, input):
        pattern = ((1, 3), (1, 5), (3, 1), (5, 1), (5, 3), (3, 5), (5, 5), (1, 1), (3, 3), (5, 5))
        for x, y in pattern:
            input[0][x][y] = 255
        return input

    def alter_data_set(self, images, targets):
        for idx, image in enumerate(images):
            images[idx] = self.add_visual_pattern(image)
            targets[idx] = 5
        return images, targets

    def update_weights(self, model, global_round, model_replacement=False, attack=None):
        # Set mode to train model
        accuracies = []
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
        self.data_size = len(self.train_loader)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                clipped_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
                images, labels = images.to(self.device), labels.to(self.device)
                if attack:
                    images, labels = self.alter_data_set(images, labels)
                model.zero_grad()
                # print(images.shape)
                log_probs = model(images)
                loss = self.criterion(log_probs.squeeze(), labels)
                loss.backward()

                    # add Gaussian noise

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
                                            100. * batch_idx / len(self.train_loader), loss))
                batch_loss.append(loss)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            sigma = compute_noise(1, 0.80, self.args.eps, self.args.E * self.args.tot_T, 0.00001, 0.1)
            for name, param in model.named_parameters():
                clipped_grads[name] += gaussian_noise(clipped_grads[name].shape, self.args.clip, sigma,
                                                      device='cuda')

                # scale back
            for name, param in model.named_parameters():
                clipped_grads[name] /= (self.data_size * 0.5)

            for name, param in model.named_parameters():
                param.grad = clipped_grads[name]
            optimizer.step()
            optimizer.zero_grad()

            accuracy, _ = self.test_inference(copy.deepcopy(model), self.test_loader)
            #print("Accuracy : ",accuracy)
            accuracies.append(accuracy)

        if model_replacement:
            return model_replacement(model.state_dict(), x, self.args.num_users, self.args), sum(epoch_loss) / len(
                epoch_loss)
        else:
            return model.state_dict(), sum(epoch_loss) / len(epoch_loss), accuracies, epoch_loss

    def update_weights_replacement(self, model, global_round, attack=False):
        # Set mode to train model
        model.train()
        epoch_loss = []
        eps = 0.1
        x = copy.deepcopy(model.state_dict())
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                if attack:
                    images, labels = self.alter_data_set(images, labels)

                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                log_probs = model(images)
                # labels = labels.type(torch.cuda.FloatTensor)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                if attack:
                    x_adv = copy.deepcopy(model.state_dict())
                    for key in x_adv.keys():
                        x_adv[key] = torch.max(torch.min(x_adv[key], x[key] + eps), x[key] - eps)
                    model.load_state_dict(x_adv)
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.train_loader.dataset),
                                            100. * batch_idx / len(self.train_loader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        if attack:
            print('test attack after attacker', attack_test_visual_pattern(self.test_dataset, model))
        if attack:
            return model_replacement(model.state_dict(), x, self.args.num_users, self.args), sum(epoch_loss) / len(
                epoch_loss)
        else:
            return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
        # return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def update_sdt_weights(self, model, global_round, model_replacement=False, attack=None):
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

        for iter in range(self.args.local_ep):
            batch_loss = []
            for inputs, targets in self.train_loader:
                inputs = inputs.to('cpu')
                targets = targets.to('cpu')

                # if training distillated tree, use NN predictions
                targets = get_model_predictions(model.model,
                                                inputs).squeeze()

                inputs = inputs.view(len(inputs), -1)
                optimizer.zero_grad()

                # training
                ones = torch.ones((len(inputs), 1)).to('cpu')
                model.forward(model.root, inputs, ones)
                loss = model.get_loss(targets)
                batch_loss.append(loss.detach().cpu().numpy())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        if model_replacement:
            return model_replacement(model.state_dict(), x, self.args.num_users, self.args), sum(epoch_loss) / len(
                epoch_loss)
        else:
            return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Inference
                outputs = model(images)
                batch_loss = self.criterion(outputs.squeeze(), labels)
                loss += batch_loss.item()

                # Prediction
                _, pred_labels = torch.max(outputs.squeeze(), 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

        accuracy = correct / total * 100
        return accuracy, loss

    def test_inference(self, model, loader):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Inference
                outputs = model(images)
                batch_loss = criterion(outputs.squeeze(), labels)
                loss += batch_loss.item()

                # Prediction
                _, pred_labels = torch.max(outputs.squeeze(), 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

        accuracy = correct / total * 100
        return accuracy, loss


    def test_sdt_inference(self, model, loader):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Inference
                outputs = model(images)
                batch_loss = self.criterion(outputs.squeeze(), labels)
                loss += batch_loss.item()

                # Prediction
                _, pred_labels = torch.max(outputs.squeeze(), 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

        accuracy = correct / total * 100
        return accuracy, loss



###################################
# Training Target and Shadow Model
###################################
def train_model(model,
                train_loader,
                val_loader,
                test_loader,
                loss,
                optimizer,
                scheduler,
                device,
                model_path,
                args,
                verbose=False,
                num_epochs=50,
                top_k=False,
                earlystopping=False,
                is_target=False, F=None,
                nbdt=False,
                sdt=False):
    best_valacc = 0
    patience = 5  # Early stopping
    stop_count = 0
    train_loss_hist = []
    valid_loss_hist = []
    val_acc_hist = []

    if is_target:
        print('----Target model training----')
    else:
        print('---Shadow model training----')

    # Path for saving best target and shadow models
    target_path = os.path.join(model_path, 'best_target_model.ckpt')
    shadow_path = os.path.join(model_path, 'best_shadow_model.ckpt')

    for epoch in range(num_epochs):

        train_loss, train_acc = train_per_epoch(model, train_loader, loss, optimizer, device)
        valid_loss, valid_acc = val_per_epoch(model, val_loader, loss, device)

        valid_loss_hist.append(valid_loss)
        train_loss_hist.append(train_loss)
        val_acc_hist.append(valid_acc)

        scheduler.step()

        print('Epoch [{}/{}], Train Loss: {:.3f} | Train Acc: {:.2f}% | Val Loss: {:.3f} | Val Acc: {:.2f}%'
              .format(epoch + 1, num_epochs, train_loss, train_acc * 100, valid_loss, valid_acc * 100))

        if earlystopping:
            if best_valacc <= valid_acc:
                print('Saving model checkpoint')
                best_valacc = valid_acc
                # Store best model weights
                best_model = copy.deepcopy(model.state_dict())
                if is_target:
                    torch.save(best_model, target_path)
                else:
                    torch.save(best_model, shadow_path)
                stop_count = 0
            else:
                stop_count += 1
                if stop_count >= patience:  # early stopping check
                    print('End Training after [{}] Epochs'.format(epoch + 1))
                    break
        else:  # Continue model training for all epochs
            print('Saving model checkpoint')
            best_valacc = valid_acc
            # Store best model weights
            best_model = copy.deepcopy(model.state_dict())
            if is_target:
                torch.save(best_model, target_path)
            else:
                torch.save(best_model, shadow_path)

    print(shadow_path)
    if is_target:
        print('----Target model training finished----')
        print('Validation Accuracy for the Target Model is: {:.2f} %'.format(100 * best_valacc))
    else:
        print('----Shadow model training finished-----')
        print('Validation Accuracy for the Shadow Model is: {:.2f} %'.format(100 * best_valacc))

    if is_target:
        print('----LOADING the best Target model for Test----')
        model.load_state_dict(torch.load(target_path))
    else:
        print('----LOADING the best Shadow model for Test----')
        model.load_state_dict(torch.load(shadow_path))

    # As the model is fully trained, time to prepare data for attack model.
    # Training Data for members would come from shadow train dataset, and member inference from target train dataset respectively.
    attack_X, attack_Y = prepare_attack_data(model, train_loader, device, top_k)

    # In test phase, we don't need to compute gradients (for memory efficiency)
    print('----Test the Trained Network----')
    if nbdt == True:
        model = SoftNBDT(model=model, dataset=nbdt_datasets_mapping[args.dataset],
                         hierarchy='target_induced_{}'.format(nbdt_models_mapping[args.classifier]))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device, dtype=torch.float)
            labels = torch.tensor([int(i) for i in labels]).to(device)

            test_outputs = model(inputs).squeeze()

            # Predictions for accuracy calculations
            _, predicted = torch.max(test_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Posterior and labels for non-members
            probs_test = Func.softmax(test_outputs, dim=-1)
            if top_k:
                # Take top K posteriors ranked high ---> low
                topk_t_probs, _ = torch.topk(probs_test, 3, dim=1)
                attack_X.append(topk_t_probs.cpu())
            else:
                attack_X.append(probs_test.cpu())
            attack_Y.append(torch.zeros(probs_test.size(0), dtype=torch.long))

        if is_target:
            print('Test Accuracy of the Target model: {:.2f}%'.format(100 * correct / total))
        else:
            print('Test Accuracy of the Shadow model: {:.2f}%'.format(100 * correct / total))


    if sdt == True :

        hidden_sizes = [32, 64, 128]
        img_size = 64
        batch_size = 64
        lr = 1e-3
        lmbda = 0.6
        weight_decay = 5e-4
        nb_epochs = 20
        depth = 7
        feature_size = args.feature_size

        model = SoftDecisionTree(model, depth, args.num_classes, feature_size, lr,
                                            weight_decay, lmbda=lmbda)

        model.is_distilled = True
        model, accuracy = train_soft_tree(model, nb_epochs, train_loader,
                                                     test_loader, test_loader)

        model = detach_soft_decision_tree(model)

        distilled_shadow_X_train = model.make_predictions(train_loader)
        distilled_shadow_X_test = model.make_predictions(test_loader)

        attack_X = [torch.cat((distilled_shadow_X_train, distilled_shadow_X_test), 0)]

    return attack_X, attack_Y


def train_per_epoch(model,
                    train_iterator,
                    criterion,
                    optimizer,
                    device,
                    bce_loss=False):
    epoch_loss = 0
    epoch_acc = 0
    correct = 0
    total = 0

    model.train()
    for _, (features, target) in enumerate(train_iterator):
        # Move tensors to the configured device
        features = features.to(device)
        # features = features.unsqueeze(1)
        target = target.to(device)

        # Forward pass
        outputs = model(features).squeeze()
        # print("OK")
        if bce_loss:
            # For BCE loss
            loss = criterion(outputs, target.unsqueeze(1))
        else:
            loss = criterion(outputs, target)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record Loss
        epoch_loss += loss.item()

        # Get predictions for accuracy calculation
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    # Per epoch valdication accuracy calculation
    epoch_acc = correct / total
    epoch_loss = epoch_loss / total

    return epoch_loss, epoch_acc

def val_per_epoch(model,
                  val_iterator,
                  criterion,
                  device,
                  bce_loss=False):
    epoch_loss = 0
    epoch_acc = 0
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for _, (features, target) in enumerate(val_iterator):
            features = features.to(device)
            target = target.to(device)

            outputs = model(features).squeeze()
            # Caluclate the loss
            if bce_loss:
                # For BCE loss
                loss = criterion(outputs, target.unsqueeze(1))
            else:
                loss = criterion(outputs, target)

            # record the loss
            epoch_loss += loss.item()

            # Check Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        # Per epoch valdication accuracy and loss calculation
        epoch_acc = correct / total
        epoch_loss = epoch_loss / total

    return epoch_loss, epoch_acc




###############################
# Training Attack Model
###############################
def train_attack_model(model,
                       dataset,
                       criterion,
                       optimizer,
                       lr_scheduler,
                       device,
                       model_path='./model',
                       epochs=10,
                       b_size=1024,
                       num_workers=1,
                       verbose=False,
                       earlystopping=False):
    n_validation = 300 # number of validation samples
    best_valacc = 0
    stop_count = 0
    patience = 5  # Early stopping

    path = os.path.join(model_path, 'best_attack_model.ckpt')

    train_loss_hist = []
    valid_loss_hist = []
    val_acc_hist = []

    train_X, train_Y = dataset

    train_X = torch.cat(train_X)
    train_Y = torch.cat(train_Y)

    # Contacetnae list of tensors to a single tensor


    # #Create Attack Dataset
    attackdataset = TensorDataset(train_X, train_Y)

    print('Shape of Attack Feature Data : {}'.format(train_X.shape))
    print('Shape of Attack Target Data : {}'.format(train_Y.shape))
    print('Length of Attack Model train dataset : [{}]'.format(len(attackdataset)))
    print('Epochs [{}] and Batch size [{}] for Attack Model training'.format(epochs, b_size))

    # Create Train and Validation Split
    n_train_samples = len(attackdataset) - n_validation
    train_data, val_data = torch.utils.data.random_split(attackdataset,
                                                         [n_train_samples, n_validation])

    b_size = 1024
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=b_size,
                                               shuffle=True,
                                               num_workers=num_workers)

    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                             batch_size=b_size,
                                             shuffle=False,
                                             num_workers=num_workers)

    print('----Attack Model Training------')
    for i in range(epochs):

        train_loss, train_acc = train_per_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = val_per_epoch(model, val_loader, criterion, device)

        valid_loss_hist.append(valid_loss)
        train_loss_hist.append(train_loss)
        val_acc_hist.append(valid_acc)

        # lr_scheduler.step()

        print('Epoch [{}/{}], Train Loss: {:.3f} | Train Acc: {:.2f}% | Val Loss: {:.3f} | Val Acc: {:.2f}%'
              .format(i + 1, epochs, train_loss, train_acc * 100, valid_loss, valid_acc * 100))

        if earlystopping:
            if best_valacc <= valid_acc:
                print('Saving model checkpoint')
                best_valacc = valid_acc
                # Store best model weights
                best_model = copy.deepcopy(model.state_dict())
                torch.save(best_model, path)
                stop_count = 0
            else:
                stop_count += 1
                if stop_count >= patience:  # early stopping check
                    print('End Training after [{}] Epochs'.format(epochs + 1))
                    break
        else:  # Continue model training for all epochs
            print('Saving model checkpoint')
            best_valacc = valid_acc
            # Store best model weights
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, path)

    return best_valacc


# Prepare data for Attack Model
def prepare_attack_data(model,
                        iterator,
                        device,
                        top_k=False,
                        test_dataset=False,
                        attack=False,
                        pastel=True,
                        layers=[]):
    attack_x = []
    attack_y = []
    model.load_state_dict(model.state_dict())
    model.eval()
    with torch.no_grad():
        for inputs, _ in iterator:
            # Move tensors to the configured device
            inputs = inputs.to(device, dtype=torch.float)

            # Forward pass through the model
            outputs = model(inputs).squeeze()

            # To get class probabilities
            posteriors = Func.softmax(outputs, dim=1)
            if top_k:
                # Top 3 posterior probabilities(high to low) for train samples
                topk_probs, _ = torch.topk(posteriors, 3, dim=1)
                attack_x.append(topk_probs.cpu())
            else:
                attack_x.append(posteriors.cpu())

            # This function was initially designed to calculate posterior for training loader,
            # but to handle the scenario when trained model is given to us, we added this boolean
            # to different if the dataset passed is training or test and assign labels accordingly
            if test_dataset:
                attack_y.append(torch.zeros(posteriors.size(0), dtype=torch.long))
            else:
                attack_y.append(torch.ones(posteriors.size(0), dtype=torch.long))

    return attack_x, attack_y
