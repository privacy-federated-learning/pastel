import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from data.motionsense.load_data import create_dataset,  load_motion_sense_item, motionsense_collate_fn
from data.speech_commands.load_data import speech_commands_collate_fn
from data.celeba.load_data import celeba_collate_fn
from data.purchase.purchase import purchase_collate_fn
from data.speech_commands.load_data import SubsetSC
import copy
import random
import json

from torch.utils.data import TensorDataset

########################
# Model Hyperparameters
########################
# Number of filters for target and shadow models
target_filters = [128, 256, 256]
shadow_filters = [64, 128, 128]
# New FC layers size for pretrained model
n_fc= [256, 128]
# For CIFAR-10 and MNIST dataset
num_classes = 10
# No. of training epocs
num_epochs = 30
# how many samples per batch to load
batch_size = 512
# learning rate
learning_rate = 0.001
# Learning rate decay
lr_decay = 0.96
# Regularizer
reg=1e-4
# percentage of dataset to use for shadow model
shadow_split = 0.6
# Number of validation samples
n_validation = 100
# Number of processes
num_workers = 2
# Hidden units for MNIST model
n_hidden_mnist = 32


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = './data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
            test_user_groups = cifar_iid(test_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            elif args.dirichlet:
                user_groups = get_distribution_index(args.alpha, train_dataset, args.dataset,  args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)


    elif args.dataset == 'cifar100':
        data_dir = './data/cifar100/'
        apply_transform = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                         transform=apply_transform)

        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                        transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
            test_user_groups = cifar_iid(test_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            elif args.dirichlet:
                user_groups = get_distribution_index(args.alpha, train_dataset, args.dataset,  args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)



    elif args.dataset == 'gtsrb':

        data_transforms = transforms.Compose([
            transforms.Resize([112, 112]),
            transforms.ToTensor()
        ])

        train_data_path = "./data/GTSRB/Train"
        train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=data_transforms)

        # Divide data into training and validation (0.8 and 0.2)
        ratio = 0.9
        n_train_examples = int(len(train_data) * ratio)
        n_val_examples = len(train_data) - n_train_examples

        train_dataset, test_dataset = torch.utils.data.random_split(train_data, [n_train_examples, n_val_examples])
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
            test_user_groups = cifar_iid(test_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            elif args.dirichlet:
                user_groups = get_distribution_index(args.alpha, train_dataset, args.dataset,  args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)


    elif args.dataset == 'purchase':
        train_dataset = TensorDataset(torch.Tensor(np.load('./data/purchase/purchase_train_data.npy')),
                                      torch.Tensor(np.load('./data/purchase/purchase_train_labels.npy')).to(torch.long))

        test_dataset = TensorDataset(torch.Tensor(np.load('./data/purchase/purchase_test_data.npy')),
                                     torch.Tensor(np.load('./data/purchase/purchase_test_labels.npy')).to(torch.long))

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
            test_user_groups = cifar_iid(test_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            elif args.dirichlet:
                user_groups = get_distribution_index(args.alpha, train_dataset, args.dataset,  args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)


    elif args.dataset == 'texas':
        train_dataset = TensorDataset(torch.Tensor(np.load('./data/texas/texas_train_data.npy')),
                                      torch.Tensor(np.load('./data/texas/texas_train_labels.npy')).to(torch.long))

        test_dataset = TensorDataset(torch.Tensor(np.load('./data/texas/texas_test_data.npy')),
                                     torch.Tensor(np.load('./data/texas/texas_test_labels.npy')).to(torch.long))

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
            test_user_groups = cifar_iid(test_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            elif args.dirichlet:
                user_groups = get_distribution_index(args.alpha, train_dataset, args.dataset,  args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'motionsense':
        train_dataset = create_dataset(root=args.motionsense_train_path, load_item_fn=load_motion_sense_item, ext_audio='.mat')
        test_dataset = create_dataset(root=args.motionsense_test_path, load_item_fn=load_motion_sense_item, ext_audio='.mat')

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
            test_user_groups = cifar_iid(test_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            elif args.dirichlet:
                user_groups = get_distribution_index(args.alpha, train_dataset, args.dataset,  args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'speech_commands':
        train_dataset = SubsetSC("training")
        test_dataset = SubsetSC("testing")
        val_dataset = SubsetSC("validation")

        user_groups = mnist_iid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        if args.dataset == 'mnist':

            data_dir = '../data/mnist/'
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)
        else:
            apply_transform = transforms.Compose([
                # transforms.Resize((227, 227)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            data_dir = '../data/fmnist/'
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                                  transform=apply_transform)

            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                                 transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            print('hey this is iid')
            user_groups = mnist_iid(train_dataset, args.num_users)
            test_user_groups = mnist_iid(test_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                print('hey this is unequal')
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            elif args.dirichlet:
                print('hey this is dirichlet')
                user_groups = get_distribution_index(args.alpha, train_dataset, args.dataset,  args.num_users)
            else:
                # Chose euqal splits for every user
                print('hey this is equal')
                user_groups = mnist_noniid(train_dataset, args.num_users)

    elif args.dataset == 'celeba':

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset = torchvision.datasets.CelebA('./data/',
                                                        download=True,
                                                        split='train',
                                                        transform=transform,
                                                        target_type="attr")

        test_dataset = torchvision.datasets.CelebA('./data/',
                                                       download=True,
                                                       split='test',
                                                       transform=transform,
                                                       target_type="attr")


        train_dataset = filter_celeba_by_indices(train_dataset, random.sample(list(range(len(train_dataset))), 30000))
        test_dataset = filter_celeba_by_indices(test_dataset, random.sample(list(range(len(test_dataset))), 10000))
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
            test_user_groups = cifar_iid(test_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            elif args.dirichlet:
                user_groups = get_distribution_index(args.alpha, train_dataset, args.dataset,  args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)
                test_user_groups = cifar_iid(test_dataset, args.num_users)


    elif args.dataset == 'imagenet':
        print('hey --')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        data_dir = '../data/imagenet/'
        apply_transform = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.ImageNet(data_dir, train=True, download=True,
                                          transform=transform)

        test_dataset = datasets.ImageNet(data_dir, train=False, download=True,
                                         transform=transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
            test_user_groups = cifar_iid(test_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            elif args.dirichlet:
                user_groups = get_distribution_index(args.alpha, train_dataset, args.dataset,  args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups



def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 2
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard + 1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        # np.random.seed(2)
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.train_labels)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        if isinstance(self.dataset, SubsetSC):

            waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[self.idxs[item]]

            image, label = (waveform, label)

            return waveform, label

        else:
            image, label = self.dataset[self.idxs[item]]
            return torch.tensor(image), torch.tensor(label)



class SpeechCommandsWrapper(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.idxs = [int(i) for i in range(len(dataset))]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

            waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[self.idxs[item]]

            image, label = (waveform, label)

            return waveform, label



## data distriution

def normalize(s):
    for idx, line in enumerate(s):
        s[idx] = line / sum(line) * 5000
    return np.cumsum(np.around(s, 0), axis=1)


def find_idx(ligne, index):
    for idx, e in enumerate(ligne):
        if index <= sum(ligne[0:int(idx) + 1]):
            break
    return idx


def somme(matrix, idx):
    tab = 0
    for i in matrix[0:idx]:
        tab = tab + i
    return tab


def get_distribution_index(alpha, dataset, dataset_name, nb_user=5):
    indiv_list = []
    nb_class = 36

    nb_user = nb_user + 20 # Offset nb of user to make sure all participants have > 0 datapoints

    if dataset_name == 'speech_commands':
        classes = list(set(datapoint[2] for datapoint in dataset))
        classes_mapping = {}
        for c in range(len(classes)):
            classes_mapping[classes[c]]=c

        labels = [classes_mapping[x] for x in list(datapoint[2] for datapoint in dataset)]

    elif dataset_name == 'celeba' :

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=16,
                                             collate_fn=celeba_collate_fn,
                                             num_workers=2)
        labels = []
        for batch_idx, (images, b_labels) in enumerate(loader):
            labels+=b_labels.tolist()

    else :
        print("Dirichlet is not implemented yet for this dataset")

    for x in range(len(labels)):
        labels[x] = (x, labels[x])

    for goal in range(0, nb_class):
        list_1 = [idx for idx, x in labels if x == goal]
        indiv_list.append(list_1)
    s = normalize(np.random.dirichlet([alpha] * nb_class, nb_user).transpose())
    data_list_transfer = []
    for user in range(0, nb_user):
        if user == 0:
            bound_1 = 0
            bound_2 = int(s[0][user])
            tmp = indiv_list[0][bound_1:bound_2]
            data_list_transfer.append(tmp)
        else:
            bound_1 = int(s[0][user - 1])
            bound_2 = int(s[0][user])
            tmp = indiv_list[0][bound_1:bound_2]
            data_list_transfer.append(tmp)

        for class_ in range(1, nb_class):
            if user == 0:
                bound_1 = 0
                bound_2 = int(s[class_][user])
                tmp = indiv_list[class_][bound_1:bound_2]
                data_list_transfer[user] = data_list_transfer[user] + tmp
            else:
                bound_1 = int(s[class_][user - 1])
                bound_2 = int(s[class_][user])
                tmp = indiv_list[class_][bound_1:bound_2]
                data_list_transfer[user] = data_list_transfer[user] + tmp

    dict_users = {}
    for i in range(nb_user):
        dict_users[i] = set(data_list_transfer[i])

    return dict_users


def get_data_transforms(dataset, augm=False):
    if dataset == 'cifar':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        test_transforms = transforms.Compose([transforms.ToTensor(),
                                              normalize])

        if augm:
            train_transforms = transforms.Compose([transforms.RandomRotation(5),
                                                   transforms.RandomHorizontalFlip(p=0.5),
                                                   transforms.RandomCrop(32, padding=4),
                                                   transforms.ToTensor(),
                                                   normalize])
        else:
            train_transforms = transforms.Compose([transforms.ToTensor(),
                                                   normalize])

    else:
        # The values 0.1307 and 0.3081 used for the Normalize() transformation below are the global mean and standard deviation
        # of the MNIST dataset
        test_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))])
        if augm:
            train_transforms = torchvision.transforms.Compose([transforms.RandomRotation(5),
                                                               transforms.RandomHorizontalFlip(p=0.5),
                                                               torchvision.transforms.ToTensor(),
                                                               torchvision.transforms.Normalize((0.1307,), (0.3081,))])
        else:

            train_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                               torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    return train_transforms, test_transforms


def split_dataset(train_dataset):
    # For simplicity we are only using orignal training set and splitting into 4 equal parts
    # and assign it to Target train/test and Shadow train/test.
    total_size = len(train_dataset)
    split1 = total_size // 4
    split2 = split1 * 2
    split3 = split1 * 3

    indices = list(range(total_size))
    np.random.seed(4)
    np.random.shuffle(indices)

    # Shadow model train and test set
    s_train_idx = indices[:split1]
    s_test_idx = indices[split1:split2]

    # Target model train and test set
    t_train_idx = indices[split2:split3]
    t_test_idx = indices[split3:]

    return s_train_idx, s_test_idx, t_train_idx, t_test_idx


def filter_motionsense_by_indices(dataset, indices):
    filtered_dataset = copy.deepcopy(dataset)
    filtered_dataset._walker = [filtered_dataset._walker[i] for i in indices]
    return filtered_dataset


def filter_cifar_by_indices(dataset, indices):
    filtered_dataset = copy.deepcopy(dataset)
    filtered_dataset.data= [filtered_dataset.data[i] for i in indices]
    return filtered_dataset

def filter_gtsrb_by_indices(dataset, indices):
    filtered_dataset = copy.deepcopy(dataset)
    filtered_dataset.indices = indices
    return filtered_dataset


def filter_purchase_by_indices(dataset, indices):
    filtered_dataset = copy.deepcopy(dataset)
    filtered_dataset.tensors = dataset[indices]
    return filtered_dataset


def filter_celeba_by_indices(dataset,indices):
    filtered_dataset = copy.deepcopy(dataset)
    filtered_dataset.attr = torch.stack([filtered_dataset.attr[i] for i in indices])
    filtered_dataset.bbox = torch.stack([filtered_dataset.bbox[i] for i in indices])
    filtered_dataset.filename = [filtered_dataset.filename[i] for i in indices]
    filtered_dataset.identity = torch.stack([filtered_dataset.identity[i] for i in indices])
    filtered_dataset.landmarks_align = torch.stack([filtered_dataset.landmarks_align[i] for i in indices])

    return filtered_dataset



def split_target_shadow_dataset(dataset, args, shadow_split=0.5, ):
    if args.dataset in ['cifar', 'cifar100'] :
        s_train_idx, s_out_idx, t_train_idx, t_out_idx = split_dataset(dataset)
        t_val_idx = t_out_idx[:n_validation]
        s_val_idx = s_out_idx[:n_validation]

        s_train_data = filter_cifar_by_indices(dataset, s_train_idx)
        s_out_data = filter_cifar_by_indices(dataset, s_out_idx)
        s_val_data = filter_cifar_by_indices(dataset, s_val_idx)

        t_train_data = filter_cifar_by_indices(dataset, t_train_idx)
        t_out_data = filter_cifar_by_indices(dataset, t_out_idx)
        t_val_data = filter_cifar_by_indices(dataset, t_val_idx)


    elif args.dataset ==  'gtsrb' :
        s_train_idx, s_out_idx, t_train_idx, t_out_idx = split_dataset(dataset)
        t_val_idx = t_out_idx[:n_validation]
        s_val_idx = s_out_idx[:n_validation]

        s_train_data = filter_gtsrb_by_indices(dataset, s_train_idx)
        s_out_data = filter_gtsrb_by_indices(dataset, s_out_idx)
        s_val_data = filter_gtsrb_by_indices(dataset, s_val_idx)

        t_train_data = filter_gtsrb_by_indices(dataset, t_train_idx)
        t_out_data = filter_gtsrb_by_indices(dataset, t_out_idx)
        t_val_data = filter_gtsrb_by_indices(dataset, t_val_idx)

    elif args.dataset in ['purchase', 'texas']:
        s_train_idx, s_out_idx, t_train_idx, t_out_idx = split_dataset(dataset)
        t_val_idx = t_out_idx[:n_validation]
        s_val_idx = s_out_idx[:n_validation]

        s_train_data = filter_purchase_by_indices(dataset, s_train_idx)
        s_out_data = filter_purchase_by_indices(dataset, s_out_idx)
        s_val_data = filter_purchase_by_indices(dataset, s_val_idx)

        t_train_data = filter_purchase_by_indices(dataset, t_train_idx)
        t_out_data = filter_purchase_by_indices(dataset, t_out_idx)
        t_val_data = filter_purchase_by_indices(dataset, t_val_idx)
        # sample training data amongs

    elif args.dataset == 'celeba':

        s_train_idx, s_out_idx, t_train_idx, t_out_idx = split_dataset(dataset)
        t_val_idx = t_out_idx[:n_validation]
        s_val_idx = s_out_idx[:n_validation]

        s_train_data = filter_celeba_by_indices(dataset, s_train_idx)
        s_out_data = filter_celeba_by_indices(dataset, s_out_idx)
        s_val_data = filter_celeba_by_indices(dataset, s_val_idx)

        t_train_data = filter_celeba_by_indices(dataset, t_train_idx)
        t_out_data = filter_celeba_by_indices(dataset, t_out_idx)
        t_val_data = filter_celeba_by_indices(dataset, t_val_idx)



    elif args.dataset == 'motionsense':

        s_train_idx, s_out_idx, t_train_idx, t_out_idx = split_dataset(dataset)
        t_val_idx = t_out_idx[:n_validation]
        s_val_idx = s_out_idx[:n_validation]

        s_train_data = filter_motionsense_by_indices(dataset, s_train_idx)
        s_out_data = filter_motionsense_by_indices(dataset, s_out_idx)
        s_val_data = filter_motionsense_by_indices(dataset, s_val_idx)

        t_train_data = filter_motionsense_by_indices(dataset, t_train_idx)
        t_out_data = filter_motionsense_by_indices(dataset, t_out_idx)
        t_val_data = filter_motionsense_by_indices(dataset, t_val_idx)



    elif args.dataset == 'speech_commands':

        s_train_idx, s_out_idx, t_train_idx, t_out_idx = split_dataset(dataset)
        random.seed(4)
        s_train_idx = random.sample(s_train_idx, 4000)
        s_out_idx = random.sample(s_out_idx, 4000)

        t_val_idx = t_out_idx[:n_validation]
        s_val_idx = s_out_idx[:n_validation]

        s_train_data = filter_motionsense_by_indices(dataset, s_train_idx)
        s_out_data = filter_motionsense_by_indices(dataset, s_out_idx)
        s_val_data = filter_motionsense_by_indices(dataset, s_val_idx)

        t_train_data = filter_motionsense_by_indices(dataset, t_train_idx)
        t_out_data = filter_motionsense_by_indices(dataset, t_out_idx)
        t_val_data = filter_motionsense_by_indices(dataset, t_val_idx)


    # sample training data amongst users
    if args.iid:
        # Sample IID user data from Mnist
        user_groups = cifar_iid(t_train_data, args.num_users)
    else:
        # Sample Non-IID user data from Mnist
        if args.unequal:
            # Chose uneuqal splits for every user
            raise NotImplementedError()
        elif args.dirichlet:
            user_groups = get_distribution_index(args.alpha, t_train_data, args.dataset, args.num_users)
        else:
            # Chose euqal splits for every user
            user_groups = cifar_noniid(t_train_data, args.num_users)

    return t_train_data, t_out_data, t_val_data, s_train_data, s_out_data, s_val_data, user_groups


def data_to_loader(data, args):

    if args.dataset == 'speech_commands':
        collate_fn = speech_commands_collate_fn
        loader = torch.utils.data.DataLoader(dataset=DatasetSplit(data, list(range(len(data)))),
                                             batch_size=args.local_bs,
                                             collate_fn=collate_fn,
                                             num_workers=num_workers)

    elif args.dataset == 'motionsense':
        collate_fn = motionsense_collate_fn
        loader = torch.utils.data.DataLoader(dataset=data,
                                             batch_size=args.local_bs,
                                             collate_fn=collate_fn,
                                             num_workers=num_workers)
    elif args.dataset == 'celeba':
        collate_fn = celeba_collate_fn
        loader = torch.utils.data.DataLoader(dataset=data,
                                             batch_size=args.local_bs,
                                             collate_fn=collate_fn,
                                             num_workers=num_workers)

    else :
        collate_fn = None

        loader = torch.utils.data.DataLoader(dataset=data,
                                             batch_size=args.local_bs,
                                             shuffle=True,
                                             collate_fn=collate_fn,
                                             num_workers=num_workers)



    return loader




def get_data_loader(dataset,
                    data_dir,
                    batch,
                    shadow_split=0.5,
                    augm_required=False,
                    num_workers=1):
    """
     Utility function for loading and returning train and valid
     iterators over the CIFAR-10 and MNIST dataset.
    """
    error_msg = "[!] shadow_split should be in the range [0, 1]."
    assert ((shadow_split >= 0) and (shadow_split <= 1)), error_msg

    train_transforms, test_transforms = get_data_transforms(dataset, augm_required)

    # Download test and train dataset
    if dataset == 'cifar':
        # CIFAR10 training set
        train_set = torchvision.datasets.CIFAR10(root=data_dir,
                                                 train=True,
                                                 transform=train_transforms,
                                                 download=True)
        # CIFAR10 test set
        test_set = torchvision.datasets.CIFAR10(root=data_dir,
                                                train=False,
                                                transform=test_transforms)

        s_train_idx, s_out_idx, t_train_idx, t_out_idx = split_dataset(train_set)
    elif dataset == 'mnist':
        # MNIST train set
        train_set = torchvision.datasets.MNIST(root=data_dir,
                                               train=True,
                                               transform=train_transforms,
                                               download=True)
        # MNIST test set
        test_set = torchvision.datasets.MNIST(root=data_dir,
                                              train=False,
                                              transform=test_transforms)

        s_train_idx, s_out_idx, t_train_idx, t_out_idx = split_dataset(train_set)

    elif dataset == 'motionsense':
        train_set = create_dataset(root='./data/motionsense/train/', load_item_fn=load_motion_sense_item, ext_audio='.mat')
        test_set = create_dataset(root='./data/motionsense/test/', load_item_fn=load_motion_sense_item, ext_audio='.mat')

        s_train_idx, s_out_idx, t_train_idx, t_out_idx = split_dataset(train_set)

    elif dataset == 'speech_commands':
        train_set = SubsetSC("training")
        test_set = SubsetSC("testing")

        s_train_idx, s_out_idx, t_train_idx, t_out_idx = split_dataset(train_set)


    # Data samplers
    s_train_sampler = SubsetRandomSampler(s_train_idx)
    s_out_sampler = SubsetRandomSampler(s_out_idx)
    t_train_sampler = SubsetRandomSampler(t_train_idx)
    t_out_sampler = SubsetRandomSampler(t_out_idx)

    # In our implementation we are keeping validation set to measure training performance
    # From the held out set for target and shadow, we take n_validation samples.
    # As train set is already small we decided to take valid samples from held out set
    # as these are samples not used in training.
    if dataset == 'cifar':
        target_val_idx = t_out_idx[:n_validation]
        shadow_val_idx = s_out_idx[:n_validation]

        t_val_sampler = SubsetRandomSampler(target_val_idx)
        s_val_sampler = SubsetRandomSampler(shadow_val_idx)
    else:
        target_val_idx = t_out_idx[:n_validation]
        shadow_val_idx = s_out_idx[:n_validation]

        t_val_sampler = SubsetRandomSampler(target_val_idx)
        s_val_sampler = SubsetRandomSampler(shadow_val_idx)

    # -------------------------------------------------
    # Data loader
    # -------------------------------------------------
    if dataset == 'cifar':

        t_train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                     batch_size=batch,
                                                     sampler=t_train_sampler,
                                                     num_workers=num_workers)

        t_out_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=batch,
                                                   sampler=t_out_sampler,
                                                   num_workers=num_workers)

        t_val_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=batch,
                                                   sampler=t_val_sampler,
                                                   num_workers=num_workers)

        s_train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                     batch_size=batch,
                                                     sampler=s_train_sampler,
                                                     num_workers=num_workers)

        s_out_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=batch,
                                                   sampler=s_out_sampler,
                                                   num_workers=num_workers)

        s_val_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=batch,
                                                   sampler=s_val_sampler,
                                                   num_workers=num_workers)

    else:

        if dataset == 'motionsense':
            collate_fn = motionsense_collate_fn
        elif dataset == 'speech_commands':
            collate_fn = speech_commands_collate_fn
            train_set = SpeechCommandsWrapper(train_set)
        else:
            collate_fn = None

        t_train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                     batch_size=batch,
                                                     collate_fn = collate_fn,
                                                     sampler=t_train_sampler,
                                                     num_workers=num_workers)

        t_out_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=batch,
                                                   collate_fn = collate_fn,
                                                   sampler=t_out_sampler,
                                                   num_workers=num_workers)

        t_val_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=batch,
                                                   collate_fn = collate_fn,
                                                   sampler=t_val_sampler,
                                                   num_workers=num_workers)

        s_train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                     batch_size=batch,
                                                     collate_fn = collate_fn,
                                                     sampler=s_train_sampler,
                                                     num_workers=num_workers)

        s_out_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=batch,
                                                   collate_fn = collate_fn,
                                                   sampler=s_out_sampler,
                                                   num_workers=num_workers)

        s_val_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=batch,
                                                   collate_fn = collate_fn,
                                                   sampler=s_val_sampler,
                                                   num_workers=num_workers)

    print('Total Test samples in {} dataset : {}'.format(dataset, len(test_set)))
    print('Total Train samples in {} dataset : {}'.format(dataset, len(train_set)))
    print('Number of Target train samples: {}'.format(len(t_train_sampler)))
    print('Number of Target valid samples: {}'.format(len(t_val_sampler)))
    print('Number of Target test samples: {}'.format(len(t_out_sampler)))
    print('Number of Shadow train samples: {}'.format(len(s_train_sampler)))
    print('Number of Shadow valid samples: {}'.format(len(s_val_sampler)))
    print('Number of Shadow test samples: {}'.format(len(s_out_sampler)))

    return t_train_loader, t_val_loader, t_out_loader, s_train_loader, s_val_loader, s_out_loader

