import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict


SRM_npy = np.load('SRM_Kernels.npy')

class L2_nrom(nn.Module):
    def __init__(self,mode='l2'):
        super(L2_nrom, self).__init__()
        self.mode = mode
    def forward(self, x):
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2,3), keepdim=True)).pow(0.5)
            norm = embedding / (embedding.pow(2).mean(dim=1, keepdim=True)).pow(0.5)
        elif self.mode == 'l1':
            _x = torch.abs(x)
            embedding = _x.sum((2,3), keepdim=True)
            norm = embedding / (torch.abs(embedding).mean(dim=1, keepdim=True))
        return norm

class Sepconv(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(Sepconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1)


    def forward(self, input):

        out1 = self.conv1(input)
        out = self.conv2(out1)
        return out

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features,
                                           num_input_features,
                                           kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(num_input_features, num_input_features,
                                           kernel_size=3, stride=1, padding=1, bias=False)),

    def forward(self, prev_features):
        new_features = self.conv1(self.relu1(self.norm1(prev_features)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock_Add(nn.Module):
    def __init__(self, num_layers, num_input_features):
        super(_DenseBlock_Add, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = init_features
        for name, layer in self.named_children():
            new_features = layer(features)
            features = features + new_features
        return features



class DenseNet_Add_1(nn.Module):
    def __init__(self, num_layers=6):
        super(DenseNet_Add_1, self).__init__()

        # 高通滤波 卷积核权重初始化
        self.srm_filters_weight = nn.Parameter(torch.Tensor(30, 1, 5, 5), requires_grad=False)
        self.srm_filters_weight.data.numpy()[:] = SRM_npy

        self.features = nn.Sequential(OrderedDict([('norm0', nn.BatchNorm2d(30)), ]))
        self.features.add_module('relu0', nn.ReLU(inplace=True))

        block = _DenseBlock_Add(
            num_layers=num_layers,
            num_input_features=30,)
        self.features.add_module('denseblock%d' % 1, block)#preprocessing

        num_features = 30

        trans = _Transition(num_input_features=num_features,
                            num_output_features=32)   # BlockB
        self.features.add_module('transition%d' % 1, trans)

    def forward(self, input):
        HPF_output = F.conv2d(input, self.srm_filters_weight, stride=1, padding=2)
        output = self.features(HPF_output)
        return output


class lwenet(nn.Module):
    def __init__(self):
        super(lwenet, self).__init__()
        #preprocessing+BlockB
        self.Dense_layers  = DenseNet_Add_1(num_layers=6)


        #feature extraction
        self.layer5 = nn.Conv2d(32, 32, kernel_size=3, padding=1) #BlockC
        self.layer5_BN = nn.BatchNorm2d(32)
        self.layer5_AC = nn.ReLU()

        self.layer6 = nn.Conv2d(32, 64, kernel_size=3, padding=1)#BlockC
        self.layer6_BN = nn.BatchNorm2d(64)
        self.layer6_AC = nn.ReLU()

        self.avgpooling2 = nn.AvgPool2d(kernel_size=3, stride=2,padding=1)

        self.layer7 = nn.Conv2d(64, 64, kernel_size=3, padding=1)#BlockC
        self.layer7_BN = nn.BatchNorm2d(64)
        self.layer7_AC = nn.ReLU()

        self.layer8 = nn.Conv2d(64, 128, kernel_size=3, padding=1)#BlockC
        self.layer8_BN  = nn.BatchNorm2d(128)
        self.layer8_AC = nn.ReLU()

        self.avgpooling3 =   nn.AvgPool2d(kernel_size=3, stride=2,padding=1)

        self.layer9 = nn.Conv2d(128, 128, kernel_size=3, padding=1)#BlockC
        self.layer9_BN = nn.BatchNorm2d(128)
        self.layer9_AC = nn.ReLU()

        self.layer10 = Sepconv(128,256)#BlockD
        self.layer10_BN = nn.BatchNorm2d(256)
        self.layer10_AC =  nn.ReLU()
        #MGP
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.L2_norm = L2_nrom(mode='l2')
        self.L1_norm = L2_nrom(mode='l1')
        #classifier
        self.fc1 = nn.Linear(256*3, 2)

    def forward(self, input):
        input = input.float()
        Dense_block_out = self.Dense_layers(input)
        layer5_out = self.layer5(Dense_block_out)
        layer5_out = self.layer5_BN(layer5_out)
        layer5_out = self.layer5_AC(layer5_out)
        
        layer6_out = self.layer6(layer5_out)
        layer6_out = self.layer6_BN(layer6_out)
        layer6_out = self.layer6_AC(layer6_out)

        avg_pooling2 = self.avgpooling2(layer6_out)

        layer7_out = self.layer7(avg_pooling2)
        layer7_out = self.layer7_BN(layer7_out)
        layer7_out = self.layer7_AC(layer7_out)

        layer8_out = self.layer8(layer7_out)
        layer8_out = self.layer8_BN(layer8_out)
        layer8_out = self.layer8_AC(layer8_out)

        avg_pooling3 = self.avgpooling3(layer8_out)

        layer9_out = self.layer9(avg_pooling3)
        layer9_out = self.layer9_BN(layer9_out)
        layer9_out = self.layer9_AC(layer9_out)

        layer10_out = self.layer10(layer9_out)
        layer10_out = self.layer10_BN(layer10_out)
        layer10_out = self.layer10_AC(layer10_out)
        output_GAP = self.GAP(layer10_out)
        output_L2 = self.L2_norm(layer10_out)
        output_L1 = self.L1_norm(layer10_out)
        output_GAP = output_GAP.view( -1,256)
        output_L2 = output_L2.view( -1,256)
        output_L1 = output_L1.view(-1, 256)
        Final_feat = torch.cat([output_GAP,output_L2,output_L1],dim=-1)

        output = self.fc1(Final_feat)

        return output, layer8_out





if __name__ == '__main__':
    from torchsummary import summary
    Input = torch.randn(1, 1, 256, 256).cuda()
    net = lwenet().cuda()
    print(summary(net,(1,256,256)))