import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from config import GAUSSIAN_NOISE
from resnet import resnet20
from privacy_defense_mechanisms.adversarial_noise_layer.anl import AdvNoise
from opacus.utils import module_modification

from vgg import vgg11_bn

def size_conv(size, kernel, stride=1, padding=0):
    out = int(((size - kernel + 2 * padding) / stride) + 1)
    return out


def size_max_pool(size, kernel, stride=None, padding=0):
    if stride == None:
        stride = kernel
    out = int(((size - kernel + 2 * padding) / stride) + 1)
    return out


# Calculate in_features for FC layer in Shadow Net
def calc_feat_linear_cifar(size):
    feat = size_conv(size, 3, 1, 1)
    feat = size_max_pool(feat, 2, 2)
    feat = size_conv(feat, 3, 1, 1)
    out = size_max_pool(feat, 2, 2)
    return out


# Calculate in_features for FC layer in Shadow Net
def calc_feat_linear_mnist(size):
    feat = size_conv(size, 5, 1)
    feat = size_max_pool(feat, 2, 2)
    feat = size_conv(feat, 5, 1)
    out = size_max_pool(feat, 2, 2)
    return out


def get_models(args):

    if args.dataset in ['cifar', 'cifar100']:
        if args.classifier == 'vgg':
            linear_size = 512
            model = VGG(ppm=args.ppm, eps=args.eps, vgg_name="VGG11", linear_size=linear_size, num_classes=args.num_classes)
        else:
            fc_size = 64
            model = resnet20(ppm=args.ppm, eps=args.eps, fc_size = fc_size, num_classes=args.num_classes)

    elif args.dataset == 'purchase':
        model = PurchaseClassifier(ppm=args.ppm, eps=args.eps, hidden_sizes=args.fc_hidden_sizes)
    elif args.dataset == 'texas':
        model = TexasClassifier(ppm=args.ppm, eps=args.eps, hidden_sizes=args.fc_hidden_sizes)
    elif args.dataset == 'mnist':
        model = CNNMnist(args)
    elif args.dataset == 'motionsense':
        model = MotionSense(4)
    elif args.dataset == 'speech_commands':
        model = M18(args.ppm)
    elif args.dataset in ['celeba', 'gtsrb']:
        if args.classifier == 'vgg':
            linear_size = 512 if args.dataset == 'celeba' else 4608
            model = VGG(ppm=args.ppm, eps=args.eps, vgg_name="VGG11", linear_size=linear_size, num_classes=args.num_classes)
        else:
            fc_size = 64 if args.dataset == 'celeba' else 576
            model = resnet20(ppm=args.ppm, eps=args.eps, fc_size = fc_size, num_classes=args.num_classes)
    if args.load_state_dict:
        model.load_state_dict(torch.load(args.model_to_load))

    if args.ppm in ['ldp', 'cdp', 'pastel_dp']:
        module_modification.convert_batchnorm_modules(model)

    return model

arch = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGGNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(arch)
        self.fcs = nn.Sequential(
            nn.Linear(in_features=512 * 1 * 1, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # print(x.shape)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, arch):
        layers = []
        in_channels = self.in_channels

        for x in arch:

            if type(x) == int:

                out_channels = x
                layers += [
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1)),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                    ]

                in_channels = x

            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


_cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, ppm, num_classes, linear_size=2048, eps=2.0):
        super(VGG, self).__init__()
        self.ppm = ppm
        self.eps = eps
        self.linear_size = linear_size

        self.features = self._make_layers(_cfg[vgg_name])
        self.classifier = nn.Linear(self.linear_size, num_classes)
        if ppm in config.GAUSSIAN_CONFIG:
            self.gaussian = GaussianNoise(eps)

    # pylint: disable=W0221,E1101
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        if self.ppm in config.GAUSSIAN_CONFIG:
            out = self.gaussian(out)


        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if self.ppm in config.ADVERSARIAL_CONFIG:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x),
                               AdvNoise(eps=self.eps),
                               nn.ReLU(inplace=True)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def conv1(self, x):
        return self.features[0](x)

    def bn1(self, x):
        return self.features[:1](x)
    def conv2(self, x):
        return self.features[:4](x)
    def bn2(self, x):
        return self.features[:5](x)
    def conv3(self, x):
        return self.features[:8](x)

    def bn3(self, x):
        return self.features[:9](x)

    def conv4(self, x):
        return self.features[:11](x)
    def bn4(self, x):
        return self.features[:12](x)

    def conv5(self, x):
        return self.features[:15](x)

    def bn5(self, x):
        return self.features[:16](x)

    def conv6(self, x):
        return self.features[:18](x)

    def bn6(self, x):
        return self.features[:19](x)
    def conv7(self, x):
        return self.features[:22](x)

    def bn7(self, x):
        return self.features[:23](x)
    def conv8(self, x):
        return self.features[:25](x)

    def bn8(self, x):
        return self.features[:26](x)


class CNN(nn.Module):
    def __init__(self, input_channel=3, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128*6*6, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

#
# class AlexNet(nn.Module):
#     def __init__(self):
#         super(AlexNet, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 11, stride=4, padding=0 )
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
#         self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 2)
#         self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1)
#         self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
#         self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
#         self.fc1  = nn.Linear(in_features= 1024, out_features= 4096)
#         self.fc2  = nn.Linear(in_features= 4096, out_features= 4096)
#         self.fc3 = nn.Linear(in_features=4096 , out_features=10)
#
#
#     def forward(self,x):
#         x = F.relu(self.conv1(x))
#         x = self.maxpool(x)
#         x = F.relu(self.conv2(x))
#         x = self.maxpool(x)
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = F.relu(self.conv5(x))
#         x = self.maxpool(x)
#         x = x.reshape(x.shape[0], -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x




class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 2):
        super(AlexNet, self).__init__()

        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.convolutional(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return torch.softmax(x, 1)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)



class M18(nn.Module):
    def __init__(self, ppm=None, n_input=1, n_output=35, stride=4, n_channel=64, eps=2.0):
        super().__init__()

        self.ppm = ppm

        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4, stride=None)

        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.conv3 = nn.Conv1d(n_channel, n_channel, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(n_channel)
        self.conv4 = nn.Conv1d(n_channel, n_channel, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(n_channel)
        self.conv5 = nn.Conv1d(n_channel, n_channel, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4, stride=None)

        self.conv6 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(2 * n_channel)
        self.conv7 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm1d(2 * n_channel)
        self.conv8 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm1d(2 * n_channel)
        self.conv9 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4, stride=None)

        self.conv10 = nn.Conv1d(2 * n_channel, 4 * n_channel, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm1d(4 * n_channel)
        self.conv11 = nn.Conv1d(4 * n_channel, 4 * n_channel, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm1d(4 * n_channel)
        self.conv12 = nn.Conv1d(4 * n_channel, 4 * n_channel, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm1d(4 * n_channel)
        self.conv13 = nn.Conv1d(4 * n_channel, 4 * n_channel, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm1d(4 * n_channel)
        self.pool4 = nn.MaxPool1d(4, stride=None)

        self.conv14 = nn.Conv1d(4 * n_channel, 8 * n_channel, kernel_size=3, padding=1)
        self.bn14 = nn.BatchNorm1d(8 * n_channel)
        self.conv15 = nn.Conv1d(8 * n_channel, 8 * n_channel, kernel_size=3, padding=1)
        self.bn15 = nn.BatchNorm1d(8 * n_channel)
        self.conv16 = nn.Conv1d(8 * n_channel, 8 * n_channel, kernel_size=3, padding=1)
        self.bn16 = nn.BatchNorm1d(8 * n_channel)
        self.conv17 = nn.Conv1d(8 * n_channel, 8 * n_channel, kernel_size=3, padding=1)
        self.bn17 = nn.BatchNorm1d(8 * n_channel)

        self.fc1 = nn.Linear(8 * n_channel, n_output)

        if ppm in config.GAUSSIAN_CONFIG:
            self.gaussian = GaussianNoise(eps)


    def hidden_forward(self, x, layer):

        output = None
        x = self.conv1(x)
        output = x if layer =='conv1' else output

        x = self.bn1(x)
        output = x if layer =='bn1' else output

        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        output = x if layer =='conv2' else output

        x = self.bn2(x)
        output = x if layer =='bn2' else output

        x = F.relu(x)

        x = self.conv3(x)
        output = x if layer =='conv3' else output

        x = self.bn3(x)
        output = x if layer =='bn3' else output

        x = F.relu(x)

        x = self.conv4(x)
        output = x if layer =='conv4' else output

        x = self.bn4(x)
        output = x if layer =='bn4' else output

        x = F.relu(x)

        x = self.conv5(x)
        output = x if layer =='conv5' else output

        x = self.bn5(x)
        output = x if layer =='bn5' else output

        x = F.relu(x)

        x = self.pool2(x)

        x = self.conv6(x)
        output = x if layer =='conv6' else output

        x = self.bn6(x)
        output = x if layer =='bn6' else output

        x = F.relu(x)

        x = self.conv7(x)
        output = x if layer =='conv7' else output

        x = self.bn7(x)
        output = x if layer =='bn7' else output

        x = F.relu(x)

        x = self.conv8(x)
        output = x if layer =='conv8' else output

        x = self.bn8(x)
        output = x if layer =='bn8' else output

        x = F.relu(x)

        x = self.conv9(x)
        output = x if layer =='conv9' else output

        x = self.bn9(x)
        output = x if layer =='bn9' else output

        x = F.relu(x)

        x = self.pool3(x)

        x = self.conv10(x)
        output = x if layer =='conv10' else output

        x = self.bn10(x)
        output = x if layer =='bn10' else output

        x = F.relu(x)

        x = self.conv11(x)
        output = x if layer =='conv11' else output

        x = self.bn11(x)
        output = x if layer =='bn11' else output

        x = F.relu(x)

        x = self.conv12(x)
        output = x if layer =='conv12' else output

        x = self.bn12(x)
        output = x if layer =='bn12' else output

        x = F.relu(x)

        x = self.conv13(x)
        output = x if layer =='conv13' else output

        x = self.bn13(x)
        output = x if layer =='bn13' else output

        x = F.relu(x)

        x = self.pool4(x)

        x = self.conv14(x)
        output = x if layer =='conv14' else output

        x = self.bn14(x)
        output = x if layer =='bn14' else output

        x = F.relu(x)

        x = self.conv15(x)
        output = x if layer =='conv15' else output

        x = self.bn15(x)
        output = x if layer =='bn15' else output

        x = F.relu(x)

        x = self.conv16(x)
        output = x if layer =='conv16' else output

        x = self.bn16(x)
        output = x if layer =='bn16' else output

        x = F.relu(x)

        x = self.conv17(x)
        output = x if layer =='conv17' else output

        x = self.bn17(x)
        output = x if layer =='bn17' else output
        x = F.relu(x)


        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)

        if self.ppm in config.GAUSSIAN_CONFIG:
            x = self.gaussian(x)

        x = F.log_softmax(x, dim=2)

        return output

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.pool2(x)

        x = self.conv6(x)
        x = F.relu(self.bn6(x))
        x = self.conv7(x)
        x = F.relu(self.bn7(x))
        x = self.conv8(x)
        x = F.relu(self.bn8(x))
        x = self.conv9(x)
        x = F.relu(self.bn9(x))
        x = self.pool3(x)

        x = self.conv10(x)
        x = F.relu(self.bn10(x))
        x = self.conv11(x)
        x = F.relu(self.bn11(x))
        x = self.conv12(x)
        x = F.relu(self.bn12(x))
        x = self.conv13(x)
        x = F.relu(self.bn13(x))
        x = self.pool4(x)

        x = self.conv14(x)
        x = F.relu(self.bn14(x))
        x = self.conv15(x)
        x = F.relu(self.bn15(x))
        x = self.conv16(x)
        x = F.relu(self.bn16(x))
        x = self.conv17(x)
        x = F.relu(self.bn17(x))

        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)

        if self.ppm in config.GAUSSIAN_CONFIG:
            x = self.gaussian(x)

        x = F.log_softmax(x, dim=2)

        return x


class PurchaseClassifier(nn.Module):
    def __init__(self, ppm=None, eps=2.0,  num_classes=100, droprate=0, hidden_sizes = [1024, 512, 256, 128], attack=False):
        super(PurchaseClassifier, self).__init__()
        self.ppm = ppm
        self.features_size = 600
        self.hidden_sizes = hidden_sizes

        self.attack = attack

        layers = []

        layers.append(nn.Linear(self.features_size, self.hidden_sizes[0]))
        layers.append(nn.Tanh())

        for i in range(len(self.hidden_sizes)-1):
            layers.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]))
            layers.append(nn.Tanh())

        self.features = nn.Sequential(*layers)


        if droprate > 0:
            self.classifier = nn.Sequential(nn.Dropout(droprate),
                                            nn.Linear(self.hidden_sizes[-1], num_classes))
        else:
            self.classifier = nn.Linear(self.hidden_sizes[-1], num_classes)
        if self.ppm == 'gnl':
            self.gaussian = GaussianNoise(eps)

    def forward(self, x):
        hidden_out = self.features(x)

        if self.ppm =='gnl':
            return self.gaussian(self.classifier(hidden_out))
        elif self.ppm =='anl':
            return self.adv1(self.classifier(hidden_out))

        else:
            return self.classifier(hidden_out)

    def layer1(self,x):
        return self.features[0](x)

    def layer2(self,x):
        return self.features[:2](x)

    def layer3(self,x):
        return self.features[:4](x)

    def layer4(self,x):
        return self.features[:6](x)
    def layer5(self,x):
        return self.features[:8](x)
    def layer6(self,x):
        return self.features[:10](x)

    def compute_model_gradients(self,  data, layer='None', labels='None'):

        input, labels = data
        output = self.forward(input)
        criterion = nn.CrossEntropyLoss().to('cuda')

        loss = criterion(output.squeeze(), labels)
        loss.backward()

        if layer == 'fc1':
            return self.features[0].weight.grad
        elif layer == 'fc2':
            return self.features[2].weight.grad
        elif layer == 'fc3':
            return self.features[4].weight.grad
        elif layer == 'fc4':
            return self.features[6].weight.grad

        elif layer == 'classifier':
            return self.classifier.weight.grad

class TexasClassifier(nn.Module):
    def __init__(self, ppm=None, eps=2.0, num_classes=100, hidden_sizes=[1024, 512, 256, 128]):
        super(TexasClassifier, self).__init__()
        self.ppm = ppm
        self.features_size = 6169
        self.hidden_sizes = hidden_sizes

        layers = []

        layers.append(nn.Linear(self.features_size, self.hidden_sizes[0]))
        layers.append(nn.Tanh())

        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]))
            layers.append(nn.Tanh())

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Linear(self.hidden_sizes[-1]
                                    , num_classes)

        if self.ppm == 'gnl':
            self.gaussian = GaussianNoise(eps)


    def forward(self, x, attack_intermediate_layer=False, layer_id=None):
        hidden_out = self.features(x)

        if self.ppm =='gnl':
            return self.gaussian(self.classifier(hidden_out))
        else:
            return self.classifier(hidden_out)

    def layer1(self,x):
        return self.features[:0](x)
    def layer2(self,x):
        return self.features[:2](x)
    def layer3(self,x):
        return self.features[:4](x)
    def layer4(self,x):
        return self.features[:6](x)
    def layer5(self,x):
        return self.features[:8](x)
    def layer6(self,x):
        return self.features[:10](x)

class MotionSense(nn.Module):

    def __init__(self, nbClasses, **kwargs):
        super(MotionSense, self).__init__()

        self.nbClasses = nbClasses

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.f1 = Flatten()


        self.linear1 = nn.Linear(1152, 1000)
        self.linear2 = nn.Linear(1000, 500)
        self.linear3 = nn.Linear(500, 500)
        self.f2 = Flatten()
        self.linear4 = nn.Linear(500, self.nbClasses)


    def forward(self, input):
        input_normalised = (input - input.mean()) / input.std()
        X = input_normalised.unsqueeze(dim=1)

        X = F.relu(self.bn1(self.conv1(X)))
        X = self.maxpool1(X)
        X = F.relu(self.bn2(self.conv2(X)))
        X = self.maxpool2(X)
        X = self.f1(X)

        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = F.relu(self.linear3(X))
        X = self.f2(X)
        X = F.relu(self.linear4(X))

        X = F.log_softmax(X, dim=1)
        return X



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,ppm, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.ppm = ppm
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))
        #self.adv1 = AdvNoise(eps=0.4)
        #self.adv2 = AdvNoise(eps=0.4)

    def forward(self, x, training=False):
        if training and (self.ppm in config.ADVERSARIAL_CONFIG):
            out = F.relu(self.adv1(self.bn1(self.conv1(x))))
            out = self.adv2(self.bn2(self.conv2(out)))
            out += self.shortcut(x)
            out = F.relu(out)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ppm, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.ppm = ppm
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(2048 * block.expansion, num_classes)
        if ppm in config.GAUSSIAN_CONFIG:
            self.gaussian = GaussianNoise(2.0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.ppm, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.ppm in config.GAUSSIAN_CONFIG:
            out = self.gaussian(out)
        return out


def resnet18(ppm, num_classes):
    return ResNet(ppm,BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(ppm):
    return ResNet(ppm,BasicBlock, [3, 4, 6, 3])


def resnet50(ppm):
    return ResNet(ppm, Bottleneck, [3, 4, 6, 3])


def resnet101(ppm):
    return ResNet(ppm,Bottleneck, [3, 4, 23, 3])


def resnet152(ppm):
    return ResNet(ppm,Bottleneck, [3, 8, 36, 3])


class CNNMnist(nn.Module):
    def __init__(self, args, useGAN=False, target_label=0):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        if useGAN:
            self.fc2 = nn.Linear(50, args.num_classes + 1)
            args.num_classes = args.num_classes + 1
            self.args = args
            args.num_classes = args.num_classes - 1
        else:
            self.fc2 = nn.Linear(50, args.num_classes)
            self.args = args
        self.useGAN = useGAN
        self.target_label = target_label

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        return out


# Target/Shadow Model for MNIST
class MNISTNet(nn.Module):
    def __init__(self, input_dim, n_hidden, out_classes=10, size=28):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=input_dim, out_channels=n_hidden, kernel_size=5),
            nn.BatchNorm2d(n_hidden), # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=n_hidden, out_channels=n_hidden * 2, kernel_size=5),
            nn.BatchNorm2d(n_hidden * 2), # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))

        features = calc_feat_linear_mnist(size)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(features ** 2 * (n_hidden * 2), n_hidden * 2),
            nn.ReLU(inplace=True), nn.Linear(n_hidden * 2, out_classes))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.classifier(out)
        return out

#
# # Attack MLP Model
# class AttackMLP(nn.Module):
#     def __init__(self, input_size, hidden_size=64, out_classes=2):
#         super(AttackMLP, self).__init__()
#         self.classifier = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(inplace=True), nn.Linear(128, hidden_size),
#             nn.ReLU(inplace=True), nn.Linear(hidden_size, out_classes))
#
#     def forward(self, x):
#         out = self.classifier(x)
#         return out


# Attack MLP Model
class AttackMLP(nn.Module):
    def __init__(self, input_size, hidden_size=64, out_classes=2):
        super(AttackMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_classes)
        )

    def forward(self, x):
        out = self.classifier(x)
        return out




class AttackMIA(nn.Module):
    def __init__(self, input_dim, n_hidden, out_classes=2):
        super(AttackMIA, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=input_dim, out_channels=n_hidden, kernel_size=1),
            nn.BatchNorm2d(n_hidden), # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=n_hidden, out_channels=n_hidden * 2, kernel_size=5),
            nn.BatchNorm2d(n_hidden * 2), # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))

        features = calc_feat_linear_mnist(3)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(features ** 2 * (n_hidden * 2), n_hidden * 2),
            nn.ReLU(inplace=True), nn.Linear(n_hidden * 2, out_classes))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.classifier(out)
        return out


class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)
        return din
