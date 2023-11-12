import torch.nn as nn
import math
from privacy_defense_mechanisms.adversarial_noise_layer.anl import AdvNoise
__all__ = ['resnet', 'resnet20', 'resnet50']
import torch
import config


class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)
        return din


def conv3x3(in_channels, out_channels, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)  # same


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, ppm=None, eps=2.0):
        super(BasicBlock, self).__init__()
        self.ppm = ppm
        self.eps = eps
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if self.ppm in config.ADVERSARIAL_CONFIG:
            self.adv1 = AdvNoise(eps=self.eps)
            self.adv2 = AdvNoise(eps=self.eps)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.ppm in config.ADVERSARIAL_CONFIG:
            out = self.adv1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.ppm in config.ADVERSARIAL_CONFIG:
            out = self.adv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, ppm, block, num_blocks, num_classes=10, fc_size = 256, droprate=0, eps=2.0):
        super(ResNet, self).__init__()

        self.ppm = ppm
        self.eps = eps

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0])
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)

        if self.ppm in config.GAUSSIAN_CONFIG:
            self.gaussian = GaussianNoise(self.eps)

        if self.ppm in config.ADVERSARIAL_CONFIG:
            self.adv1 = AdvNoise(eps=self.eps)

        if droprate > 0:
            self.fc = nn.Sequential(nn.Dropout(droprate),
                                    nn.Linear(64 * block.expansion, num_classes))
        else:
            self.fc = nn.Linear(fc_size * block.expansion, num_classes)

        # initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.ppm in config.ADVERSARIAL_CONFIG:
                downsample = nn.Sequential(
                    AdvNoise(eps=self.eps),
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, ppm=self.ppm, eps = self.eps))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block( self.inplanes, planes, ppm=self.ppm, eps = self.eps))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        if self.ppm in config.ADVERSARIAL_CONFIG:
            x = self.adv1(x)
        x = self.relu(x)  # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.ppm in config.GAUSSIAN_CONFIG:
            x = self.gaussian(x)



        return x
        def forward(self, x):
            output = self.conv1(x)
            output = self.conv2_x(output)
            output = self.conv3_x(output)
            output = self.conv4_x(output)
            output = self.conv5_x(output)
            output = self.avg_pool(output)
            output = output.view(output.size(0), -1)
            output = self.fc(output)

        return output

    def forward_bn1(self,x):
        output = self.conv1(x)
        return output

    def forward_conv1(self, x):
        conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        conv1.to('cuda')
        output = conv1(x)
        return output

    def forward_bn17(self,x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        return output

    def forward_conv2(self, x):
        output = self.conv1(x)
        conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        conv2.to('cuda')
        output = conv2(output)
        return output

    def forward_bn2(self, x):
        output = self.conv1(x)
        bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        conv2.to('cuda')
        bn2.to('cuda')
        output = conv2(output)
        output = bn2(output)
        return output

    def forward_conv3(self, x):
        output = self.forward_bn2(x)
        relu = nn.ReLU(inplace=True)
        relu.to('cuda')
        output = relu(output)
        conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        conv3.to('cuda')
        output = conv3(output)
        return output

    def forward_bn3(self, x):
        output = self.forward_conv3(x)
        bn3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        bn3.to('cuda')
        output = bn3(output)
        return output

    def forward_conv4(self, x):
        output = self.forward_bn3(x)
        conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        conv4.to('cuda')
        output = conv4(output)
        return output

    def forward_bn4(self, x):
        output = self.forward_bn3(x)
        bn4 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        conv4.to('cuda')
        bn4.to('cuda')
        output = conv4(output)
        output = bn4(output)
        return output

    def forward_conv5(self, x):
        output = self.forward_bn4(x)
        relu = nn.ReLU(inplace=True)
        relu.to('cuda')
        output = relu(output)
        conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        conv3.to('cuda')
        output = conv3(output)
        return output

    def forward_bn5(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        return output

    def forward_conv6(self, x):
        output = self.forward_bn5(x)
        conv6 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        conv6.to('cuda')
        output = conv6(output)
        return output

    def forward_bn6(self, x):
        output = self.forward_conv6(x)
        bn6 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        bn6.to('cuda')
        output = bn6(output)
        return output

    def forward_conv7(self, x):
        output = self.forward_bn6(x)
        relu = nn.ReLU(inplace=True)
        relu.to('cuda')
        output = relu(output)
        conv7 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        conv7.to('cuda')
        output = conv7(output)
        return output

    def forward_bn7(self, x):
        output = self.forward_conv7(x)
        bn7 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        bn7.to('cuda')
        output = bn7(output)
        return output

    def forward_conv8(self, x):
        output = self.forward_bn7(x)
        conv8 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        conv8.to('cuda')
        output = conv8(output)
        return output

    def forward_bn8(self, x):
        output = self.forward_conv8(x)
        bn8 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        bn8.to('cuda')
        output = bn8(output)
        return output

    def forward_conv9(self, x):
        output = self.forward_bn8(x)
        conv9 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        conv9.to('cuda')
        output = conv9(output)
        return output

    def forward_bn9(self, x):
        output = self.forward_conv9(x)
        bn9 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        bn9.to('cuda')
        output = bn9(output)
        return output

    def forward_conv10(self, x):
        output = self.forward_bn9(x)
        relu = nn.ReLU(inplace=True)
        relu.to('cuda')
        output = relu(output)
        conv10 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        conv10.to('cuda')
        output = conv10(output)
        return output

    def forward_bn10(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        return output



def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)


def resnet20(**kwargs):
    return ResNet(block=BasicBlock, num_blocks=[3, 3, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(block=Bottleneck, num_blocks=[4, 8, 4], **kwargs)
