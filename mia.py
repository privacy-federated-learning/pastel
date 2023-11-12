import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import get_data_loader, batch_size, shadow_split, num_workers, shadow_filters, num_classes, num_epochs, \
    learning_rate, reg, lr_decay
from model import get_models, MNISTNet, AttackMLP, calc_feat_linear_cifar, MotionSense, AttackMIA, CNNMnist, M18, PurchaseClassifier, TexasClassifier, resnet18, VGG
from sklearn.metrics import classification_report
from torch.utils.data.dataset import TensorDataset
from train import train_model, train_attack_model
from resnet import resnet20

import sys
sys.path.append('model_transfer_exp/neural_backed_decision_trees')

from sklearn import metrics


# from motionsense_attack import PopulationAttack

need_earlystop = False

################################
# Attack Model Hyperparameters
################################
NUM_EPOCHS = 100
BATCH_SIZE = 1024
# Learning rate
LR_ATTACK = 0.01
# L2 Regulariser
REG = 1e-7
# weight decay
LR_DECAY = 0.96
# No of hidden units
n_hidden = 128
# Binary Classsifier
out_classes = 2


# Parameter Initialization
def init_params(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.zeros_(m.bias)


# Shadow Model mimicking target model architecture, for our implememtation is different than target
class ShadowNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, size, out_classes):
        super(ShadowNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_layers[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_layers[0]),
            # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_layers[0], out_channels=hidden_layers[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_layers[1]),
            # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        features = calc_feat_linear_cifar(size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((features ** 2 * hidden_layers[1]), hidden_layers[2]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layers[2], out_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.classifier(out)
        return out


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
            inputs = inputs.to(device)

            # Forward pass through the model
            outputs = model(inputs).squeeze()
            # To get class probabilities
            try:
                posteriors = F.softmax(outputs, dim=1)
            except:
                continue
            if top_k:
                # Top 3 posterior probabilities(high to low) for train samples
                topk_probs, _ = torch.topk(posteriors, 10, dim=1)
                attack_x.append(topk_probs.cpu().squeeze())
            else:
                attack_x.append(posteriors.cpu().squeeze())

            # This function was initially designed to calculate posterior for training loader,
            # but to handle the scenario when trained model is given to us, we added this boolean
            # to different if the dataset passed is training or test and assign labels accordingly
            if test_dataset:
                attack_y.append(torch.zeros(posteriors.size(0), dtype=torch.long))
            else:
                attack_y.append(torch.ones(posteriors.size(0), dtype=torch.long))



    return attack_x, attack_y


def find_percentage_agreement(s1, s2):
    assert len(s1) == len(s2), "Lists must have the same shape"
    nb_agreements = 0  # initialize counter to 0
    for idx, value in enumerate(s1):
        if s2[idx] == value:
            nb_agreements += 1

    percentage_agreement = nb_agreements / len(s1)

    return percentage_agreement


def attack_inference(model,
                     test_X,
                     test_Y,
                     device):
    print('----Attack Model Testing----')

    targetnames = ['Non-Member', 'Member']
    pred_y = []
    true_y = []

    # Tuple of tensors
    # X = torch.cat(test_X)
    # Y = torch.cat(test_Y)

    # Create Inference dataset
    inferdataset = TensorDataset(test_X, test_Y)

    dataloader = torch.utils.data.DataLoader(dataset=inferdataset,
                                             batch_size=1024,
                                             shuffle=False,
                                             num_workers=num_workers)

    # Evaluation of Attack Model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # Predictions for accuracy calculations
            _, predictions = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            # print('True Labels for Batch [{}] are : {}'.format(i,labels))
            # print('Predictions for Batch [{}] are : {}'.format(i,predictions))

            true_y.append(labels.cpu())
            pred_y.append(predictions.cpu())

    attack_acc = correct / total
    print('Attack Test Accuracy is  : {:.2f}%'.format(100 * attack_acc))

    true_y = torch.cat(true_y).numpy()
    pred_y = torch.cat(pred_y).numpy()

    print('---Detailed Results----')
    print(classification_report(true_y, pred_y, target_names=targetnames))
    fpr, tpr, thresholds = metrics.roc_curve(true_y, pred_y, pos_label=1)
    auc = metrics.auc(fpr, tpr)*100

    print("AUC : {}".format(auc))

    return 100 * attack_acc, auc


def prepare_attack_model(s_train_loader, s_test_loader, s_val_loader, args):

    if args.dataset in ['cifar', 'cifar100']:
        img_size = 32
        input_dim = 3
    else:  # MNIST
        img_size = 28
        input_dim = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)


    if args.dataset in ['cifar', 'cifar100']:
        shadow_model = ShadowNet(input_dim, shadow_filters, img_size, args.num_classes).to(device)
    elif args.dataset == 'motionsense':
        shadow_model = MotionSense(4).to(device)
    elif args.dataset == 'mnist':
        # Using less hidden units than target model to mimic the architecture
        n_shadow_hidden = 16
        shadow_model = MNISTNet(input_dim, n_shadow_hidden, num_classes).to(device)
    elif args.dataset == 'speech_commands':
        shadow_model = M18().to(device)
    elif args.dataset == 'purchase':
        shadow_model = PurchaseClassifier(hidden_sizes=args.fc_hidden_sizes).to(device)
    elif args.dataset == 'texas':
        shadow_model = TexasClassifier(hidden_sizes=args.fc_hidden_sizes).to(device)
    elif args.dataset in ['celeba', 'gtsrb']:
        if args.classifier == 'vgg':
            linear_size = 512 if args.dataset == 'celeba' else 4608
            shadow_model = VGG(ppm=None, vgg_name="VGG11", linear_size=linear_size,
                               num_classes=args.num_classes).to(device)
        else:
            fc_size = 64 if args.dataset == 'celeba' else 576
            shadow_model = resnet20(ppm=None, fc_size=fc_size, num_classes=args.num_classes).to(device)

    if args.param_init:
        # Initialize params
        shadow_model.apply(init_params)
    else:
        shadow_model.load_state_dict(torch.load('attack_model/best_shadow_model.ckpt'))

    # Print the model we just instantiated
    if args.verbose:
        print('----Shadow Model Architecure---')
        print(shadow_model)
        print('---Model Learnable Params----')
        for name, param in shadow_model.named_parameters():
            if param.requires_grad:
                print("\t", name)

    # Loss and optimizer
    if args.criterion == 'cross_entropy':
        shadow_loss = nn.CrossEntropyLoss()
    elif args.criterion == 'nll':
        shadow_loss = F.nll_loss
    elif args.dataset in ['cifar', 'cifar1OO', 'mnist', 'purchase', 'texas', 'celeba']:
        shadow_loss = nn.CrossEntropyLoss()
    elif args.dataset == 'motionsense' or args.dataset == 'speech_commands':
        shadow_loss = F.nll_loss


    shadow_optimizer = torch.optim.Adam(shadow_model.parameters(), lr=learning_rate, weight_decay=reg)
    shadow_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(shadow_optimizer, gamma=lr_decay)

    shadow_x, shadow_y = train_model(shadow_model,
                                     s_train_loader,
                                     s_val_loader,
                                     s_test_loader,
                                     shadow_loss,
                                     shadow_optimizer,
                                     shadow_lr_scheduler,
                                     device,
                                     args.attack_directory,
                                     args,
                                     args.verbose,
                                     num_epochs,
                                     args.need_topk,
                                     need_earlystop,
                                     is_target=False)


    input_size = shadow_x[0].shape[1]
    print('Input Feature dim for Attack Model : [{}]'.format(input_size))

    attack_model = AttackMLP(input_size, n_hidden, out_classes).to(device)

    if args.param_init:
        # Initialize params
        attack_model.apply(init_params)
    # else:
    #    attack_model.load_state_dict(torch.load('attack_model/best_attack_model.ckpt'))
    # Loss and optimizer
    attack_loss = nn.CrossEntropyLoss()
    attack_optimizer = torch.optim.Adam(attack_model.parameters(), lr=LR_ATTACK)
    attack_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(attack_optimizer, gamma=LR_DECAY)
    # Feature vector and labels for training Attack model
    attack_dataset = (shadow_x, shadow_y)

    attack_valacc = train_attack_model(attack_model, attack_dataset, attack_loss,
                                       attack_optimizer, attack_lr_scheduler, device, args.attack_directory,
                                       num_epochs, batch_size, num_workers, args.verbose)
    print('Validation Accuracy for the Best Attack Model is: {:.2f} %'.format(100 * attack_valacc))

def create_attack(t_train_loader, t_test_loader, t_val_loader,
                  s_train_loader, s_test_loader, s_val_loader, args, state_dict_model):
    batch_size = 256
    if args.dataset == 'cifar':
        img_size = 32
        input_dim = 3
    else:  # MNIST
        img_size = 28
        input_dim = 1

    # Create dataset and model directories
    if not os.path.exists(args.data_path):
        try:
            os.makedirs(args.data_path)
        except OSError:
            pass

    if not os.path.exists(args.attack_directory):
        try:
            os.makedirs(args.attack_directory)
        except OSError:
            pass

            # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Creating data loaders
    # t_train_loader, t_val_loader, t_test_loader, \
    # s_train_loader, s_val_loader, s_test_loader = get_data_loader(args.dataset,
    #                                                               args.data_path,
    #                                                               batch_size,
    #                                                               shadow_split,
    #                                                               args.need_augm,
    #                                                               num_workers)

    print('Use Target model at the path ====> [{}] '.format(args.model_path))
    # Instantiate Target Model Class
    target_model = get_models(args)



    target_model.load_state_dict(state_dict_model)
    target_model.to(device)

    print('---Peparing Attack Training data---')
    t_train_x, t_train_y = prepare_attack_data(target_model, t_train_loader, device, args.need_topk)
    t_test_x, t_test_y = prepare_attack_data(target_model, t_test_loader, device, args.need_topk, test_dataset=True)

    t_test_x = torch.cat(t_test_x)
    t_test_y = torch.cat(t_test_y)
    t_train_x = torch.cat(t_train_x)[:len(t_test_x)]
    t_train_y = torch.cat(t_train_y)[:len(t_test_y)]

    targetX = torch.cat((t_train_x, t_test_x), 0)
    targetY = torch.cat((t_train_y, t_test_y), 0)
    #
    # targetX = t_train_x + t_test_x
    # targetY = t_train_y + t_test_y


    if args.train_attack_model:
        if args.train_shadow_model:
            if args.dataset == 'cifar' :
                shadow_model = ShadowNet(input_dim, shadow_filters, img_size, num_classes).to(device)
            elif args.dataset == 'motionsense':
                shadow_model = MotionSense(4).to(device)
            elif args.dataset == 'mnist':
                # Using less hidden units than target model to mimic the architecture
                n_shadow_hidden = 16
                shadow_model = MNISTNet(input_dim, n_shadow_hidden, num_classes).to(device)
            elif args.dataset == 'speech_commands':
                shadow_model = M18().to(device)
            elif args.dataset == 'purchase':
                shadow_model = PurchaseClassifier(hidden_sizes=args.fc_hidden_sizes).to(device)
            elif args.dataset == 'texas':
                shadow_model = TexasClassifier(hidden_sizes=args.fc_hidden_sizes).to(device)
            elif args.dataset in ['celeba', 'gtsrb']:
                if args.classifier == 'vgg':
                    linear_size = 512 if args.dataset == 'celeba' else 4608
                    shadow_model = VGG(ppm=None, vgg_name="VGG11", linear_size=linear_size,
                                       num_classes=args.num_classes).to(device)
                else:
                    fc_size = 256 if args.dataset == 'celeba' else 576
                    shadow_model = resnet20(ppm=None, fc_size=fc_size, num_classes=args.num_classes).to(device)

            if args.param_init:
                # Initialize params
                shadow_model.apply(init_params)
            else:
                shadow_model.load_state_dict(torch.load('attack_model/best_shadow_model.ckpt'))

            # Print the model we just instantiated
            if args.verbose:
                print('----Shadow Model Architecure---')
                print(shadow_model)
                print('---Model Learnable Params----')
                for name, param in shadow_model.named_parameters():
                    if param.requires_grad:
                        print("\t", name)

            # Loss and optimizer
            if args.criterion == 'cross_entropy':
                shadow_loss =  nn.CrossEntropyLoss()
            elif args.criterion == 'nll':
                shadow_loss = F.nll_loss
            elif args.dataset in ['cifar', 'mnist', 'purchase', 'texas', 'celeba']:
                shadow_loss = nn.CrossEntropyLoss()
            elif args.dataset == 'motionsense' or args.dataset == 'speech_commands':
                shadow_loss = F.nll_loss


            shadow_optimizer = torch.optim.Adam(shadow_model.parameters(), lr=learning_rate, weight_decay=reg)
            shadow_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(shadow_optimizer, gamma=lr_decay)

            shadow_x, shadow_y = train_model(shadow_model,
                                             s_train_loader,
                                             s_val_loader,
                                             s_test_loader,
                                             shadow_loss,
                                             shadow_optimizer,
                                             shadow_lr_scheduler,
                                             device,
                                             args.attack_directory,
                                             args,
                                             args.verbose,
                                             num_epochs,
                                             args.need_topk,
                                             need_earlystop,
                                             is_target=False)
        else:  # Shadow model training not required, load the saved checkpoint
            print('Using Shadow model at the path  ====> [{}] '.format(f"{args.attack_directory}/best_shadow_model.ckpt"))
            shadow_file = './attack_model/best_shadow_model.ckpt'
            assert os.path.isfile(shadow_file), 'Shadow Mode Checkpoint not found, aborting load'
            # Instantiate Shadow Model Class
            if args.dataset == 'cifar':
                shadow_model = ShadowNet(input_dim, shadow_filters, img_size, num_classes).to(device)
            elif args.dataset == 'motionsense':
                shadow_model = MotionSense(4).to(device)
            elif args.dataset == 'speech_commands':
                shadow_model = M18().to(device)
            elif args.dataset == 'purchase':
                shadow_model = PurchaseClassifier().to(device)
            elif args.dataset == 'texas':
                shadow_model = TexasClassifier(hidden_sizes=args.fc_hidden_sizes).to(device)
            elif args.dataset in ['celeba', 'gtsrb']:
                if args.classifier == 'vgg':
                    linear_size = 512 if args.dataset == 'celeba' else 4608
                    shadow_model = VGG(ppm=None, vgg_name="VGG11", linear_size=linear_size,
                                       num_classes=args.num_classes).to(device)
                else:
                    fc_size = 256 if args.dataset == 'celeba' else 576
                    shadow_model = resnet20(ppm=None, fc_size=fc_size, num_classes=args.num_classes).to(device)
            else:
                # Using less hidden units than target model to mimic the architecture
                n_shadow_hidden = 16
                shadow_model = MNISTNet(input_dim, n_shadow_hidden, num_classes).to(device)

            # Load the saved model
            shadow_model.load_state_dict(torch.load(shadow_file))
            # Prepare dataset for training attack model

            print('----Preparing Attack training data---')
            train_x, train_y = prepare_attack_data(shadow_model, s_train_loader, device, args.need_topk)
            test_x, test_y = prepare_attack_data(shadow_model, s_test_loader, device, args.need_topk, test_dataset=True)

            shadow_x = train_x + test_x
            shadow_y = train_y + test_y
            # svm_attack_label_train = torch.cat(train_y + t_train_y, 0)
            # svm_attack_data_train = torch.cat(train_x + t_train_x, 0)
            # svm_attack_data_test = torch.cat(test_x + t_test_x, 0)
            # svm_attack_label_test = torch.cat(test_y + t_test_y, 0)
            """
            clf = svm.SVC(kernel='linear', C=1.0)
            clf.fit(svm_attack_data_train, svm_attack_label_train)
            print(find_percentage_agreement(list(clf.predict(svm_attack_data_test)), list(svm_attack_label_test)))
            """

        ###################################
        # Attack Model Training
        ##################################
        # The input dimension to MLP attack model

        input_size = targetX.shape[1]
        print('Input Feature dim for Attack Model : [{}]'.format(input_size))

        attack_model = AttackMLP(input_size, n_hidden, out_classes).to(device)

        if args.param_init:
            # Initialize params
            attack_model.apply(init_params)
        # else:
        #    attack_model.load_state_dict(torch.load('attack_model/best_attack_model.ckpt'))
        # Loss and optimizer
        attack_loss = nn.CrossEntropyLoss()
        attack_optimizer = torch.optim.Adam(attack_model.parameters(), lr=LR_ATTACK)
        attack_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(attack_optimizer, gamma=LR_DECAY)
        # Feature vector and labels for training Attack model
        attack_dataset = (shadow_x, shadow_y)

        attack_valacc = train_attack_model(attack_model, attack_dataset, attack_loss,
                                           attack_optimizer, attack_lr_scheduler, device, args.attack_directory,
                                           num_epochs, batch_size, num_workers, args.verbose)
        print('Validation Accuracy for the Best Attack Model is: {:.2f} %'.format(100 * attack_valacc))

# Load the trained attack model
    input_size = targetX.shape[1]
    print('Input Feature dim for Attack Model : [{}]'.format(input_size))
    attack_model = AttackMLP(input_size, n_hidden, out_classes).to(device)
    attack_path = os.path.join(args.attack_directory, 'best_attack_model.ckpt')
    attack_model.load_state_dict(torch.load(attack_path))

# Inference on trained attack model
    return attack_inference(attack_model, targetX, targetY, device)
