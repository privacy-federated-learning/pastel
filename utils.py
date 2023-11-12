
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from aggregation import orderdict_tolist, list_todict
from data.motionsense.load_data import motionsense_collate_fn
from data.speech_commands.load_data import speech_commands_collate_fn
from data.celeba.load_data import celeba_collate_fn

from dataset import SpeechCommandsWrapper

import torch.nn.functional as F


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'

    if args.dataset == 'motionsense':
        collate_fn = motionsense_collate_fn
    elif args.dataset == 'speech_commands':
        collate_fn = speech_commands_collate_fn
        test_dataset = SpeechCommandsWrapper(test_dataset)
    elif args.dataset == 'celeba':
        collate_fn = celeba_collate_fn
    else:
        collate_fn = None

    if args.dataset in ['motionsense', 'mnist', 'speech_commands']:
        criterion = nn.NLLLoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss()

    testloader = DataLoader(test_dataset, batch_size=args.local_bs,
                            collate_fn=collate_fn, shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            if outputs.shape[0]==1:
                continue
            if not (len(outputs.shape) == 2 and outputs.shape[0] == 1):
                outputs = outputs.squeeze()
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs.squeeze(), 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
    return accuracy, loss

def attack_test_visual_pattern(test_loader, model, device='cuda'):
    total = 0
    total_0 = 0
    testloader = DataLoader(test_loader, batch_size=100,
                            shuffle=False)
    for dataset_idx, (data, targets) in enumerate(testloader):
        # print(data.size())
        for index, data_in in enumerate(data):
            data[index] = add_visual_pattern(data_in)
        # print(data.size())
        test_result = model(data.to(device))
        for x in range(0, 100):

            #             if targets[x] == 0:
            tmp = torch.argmax(test_result[x])
            if tmp == 5:
                total_0 += 1
            total += 1

    return (total_0 / total) * 100


def test_per_class_accuracy(test_loader, model, device='cuda'):
    total = [0] * 10
    total_0 = [0] * 10
    testloader = DataLoader(test_loader, batch_size=100,
                            shuffle=False)
    for dataset_idx, (data, targets) in enumerate(testloader):
        test_result = model(data.to(device))
        for x in range(0, 100):
            tmp = torch.argmax(test_result[x])
            total[int(targets[x])] += 1
            if int(targets[x]) == int(tmp):
                total_0[int(tmp)] += 1

    return [i / j * 100 for i, j in zip(total_0, total)]


def add_visual_pattern(input):
    pattern = ((1, 3), (1, 5), (3, 1), (5, 1), (5, 3), (3, 5), (5, 5), (1, 1), (3, 3), (5, 5))
    for x, y in pattern:
        input[0][x][y] = 255
    return input


def split_dataset_by_class(dst, class_number, batch_size):
    all_datasets = []
    class_indices = []
    sizes = []
    for i in range(0, class_number):
        class_indices.append([])

    for i in range(0, len(dst.indices)):
        data = dst.dataset[i][1]
        list_class = class_indices[data]
        list_class.append(i)

    for ctr in range(0, class_number):
        class_subset = torch.utils.data.Subset(dst.dataset, class_indices[ctr])
        data_loader_subset = torch.utils.data.DataLoader(class_subset, batch_size, shuffle=False)
        all_datasets.append(data_loader_subset)
        sizes.append(len(class_subset.indices))

    return all_datasets, sizes


def model_replacement(attack_model, global_model
                      , num_users, args):
    list_global_model = orderdict_tolist(global_model)
    list_attack_model = orderdict_tolist(attack_model)
    results = list_global_model + np.subtract(list_attack_model, list_global_model) / num_users
    return list_todict(results, global_model)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)