import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


#################################
########### BRANCHING ###########
#################################

class Branching(nn.Module):

    def __init__(self, branched_modules=None):
        """

        Args:
            branch_config: A tuple containing number of branches and the output size.
        """
        # TODO: Make an auto naming function for this.

        super(Branching, self).__init__()

        """ ---------------------- BRANCHING MODULE --------------------- """
        if branched_modules is None:
            raise ValueError("No model provided after branching")

        self.branched_modules = nn.ModuleList(branched_modules)




    # TODO: iteration control should go inside the logger, somehow

    def forward(self, x):
        # get only the speeds from measurement labels



        # TODO: we could easily place this speed outside

        branches_outputs = []
        for branch in self.branched_modules:
            branches_outputs.append(branch(x))

        return branches_outputs


    def load_network(self, checkpoint):
        """
        Load a network for a given model definition .

        Args:
            checkpoint: The checkpoint that the user wants to add .



        """
        # coil_logger.add_message('Loading', {
        #             "Model": {"Loaded checkpoint: " + str(checkpoint) }

        #         })



        # TODO: implement
        raise NotImplementedError

#################################
########### JOIN ################
#################################

class Join(nn.Module):

    def __init__(self, params=None, module_name='Default'):
        # TODO:  For now the end module is a case
        # TODO: Make an auto naming function for this.

        super(Join, self).__init__()

        if params is None:
            raise ValueError("Creating a NULL fully connected block")
        if 'mode' not in params:
            raise ValueError(" Missing the mode parameter ")
        if 'after_process' not in params:
            raise ValueError(" Missing the after_process parameter ")

        """" ------------------ IMAGE MODULE ---------------- """
        # Conv2d(input channel, output channel, kernel size, stride), Xavier initialization and 0.1 bias initialization

        self.after_process = params['after_process']
        self.mode = params['mode']



    # TODO: iteration control should go inside the logger, somehow

    def forward(self, x, m):
        # get only the speeds from measurement labels

        if self.mode == 'cat':
            j = torch.cat((x, m), 1)

        else:
            raise ValueError("Mode to join networks not found")

        return self.after_process(j)




    def load_network(self, checkpoint):
        """
        Load a network for a given model definition .

        Args:
            checkpoint: The checkpoint that the user wants to add .



        """
        # coil_logger.add_message('Loading', {
        #             "Model": {"Loaded checkpoint: " + str(checkpoint) }

        #         })



        # TODO: implement
        raise NotImplementedError


#################################
########### FC ###########
#################################

class FC(nn.Module):

    def __init__(self, params=None, module_name='Default'
                 ):
        # TODO: Make an auto naming function for this.

        super(FC, self).__init__()


        """" ---------------------- FC ----------------------- """
        if params is None:
            raise ValueError("Creating a NULL fully connected block")
        if 'neurons' not in params:
            raise ValueError(" Missing the kernel sizes parameter ")
        if 'dropouts' not in params:
            raise ValueError(" Missing the dropouts parameter ")
        if 'end_layer' not in params:
            raise ValueError(" Missing the end module parameter ")

        if len(params['dropouts']) != len(params['neurons'])-1:
            raise ValueError("Dropouts should be from the len of kernels minus 1")


        self.layers = []


        for i in range(0, len(params['neurons']) -1):

            fc = nn.Linear(params['neurons'][i], params['neurons'][i+1])
            dropout = nn.Dropout2d(p=params['dropouts'][i])
            relu = nn.ReLU(inplace=True)

            if i == len(params['neurons'])-2 and params['end_layer']:
                self.layers.append(nn.Sequential(*[fc, dropout]))
            else:
                self.layers.append(nn.Sequential(*[fc, dropout, relu]))


        self.layers = nn.Sequential(*self.layers)



    def forward(self, x):
        # if X is a tuple, just return the other elements, the idea is to re pass
        # the intermediate layers for future attention plotting
        if type(x) is tuple:
            return self.layers(x[0]), x[1]
        else:
            return self.layers(x)



#################################
########### CONV ################
#################################


class Conv(nn.Module):

    def __init__(self, params=None, module_name='Default'):
        super(Conv, self).__init__()

        if params is None:
            raise ValueError("Creating a NULL fully connected block")
        if 'channels' not in params:
            raise ValueError(" Missing the channel sizes parameter ")
        if 'kernels' not in params:
            raise ValueError(" Missing the kernel sizes parameter ")
        if 'strides' not in params:
            raise ValueError(" Missing the strides parameter ")
        if 'dropouts' not in params:
            raise ValueError(" Missing the dropouts parameter ")
        if 'end_layer' not in params:
            raise ValueError(" Missing the end module parameter ")

        if len(params['dropouts']) != len(params['channels'])-1:
            raise ValueError("Dropouts should be from the len of channel_sizes minus 1")


        """" ------------------ IMAGE MODULE ---------------- """
        # Conv2d(input channel, output channel, kernel size, stride), Xavier initialization and 0.1 bias initialization


        self.layers = []

        for i in range(0, len(params['channels'])-1):
            conv = nn.Conv2d(in_channels=params['channels'][i], out_channels=params['channels'][i+1],
                             kernel_size=params['kernels'][i], stride=params['strides'][i])

            dropout = nn.Dropout2d(p=params['dropouts'][i])
            relu = nn.ReLU(inplace=True)
            bn = nn.BatchNorm2d(params['channels'][i+1])

            layer = nn.Sequential(*[conv, bn, dropout, relu])

            self.layers.append(layer)

        self.layers = nn.Sequential(*self.layers)
        self.module_name = module_name


    def forward(self, x):

        """ Each conv is: conv + batch normalization + dropout + relu """
        x = self.layers(x)

        x = x.view(-1, self.num_flat_features(x))

        return x, self.layers


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


    def get_conv_output(self, shape):
        """
           By inputing the shape of the input, simulate what is the ouputsize.
        """

        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat, _ = self.forward(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size



#################################
########### RESNET ###########
#################################


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

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
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2) # , stride=0

        # TODO: THis is a super hardcoding ..., in order to fit my image size on resnet
        if block.__name__ == 'Bottleneck':
            self.fc = nn.Linear(6144, num_classes)
        else:
            self.fc = nn.Linear(1536, num_classes) # this one

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # print(f'{x4.shape=}')

        x = self.avgpool(x4)
        x = x.view(x.size(0), -1)
        # print(f'{x.shape=}')
        # x = self.fc(x) # cooment this out for feature extraction for carla d4rl

        return x, [x0, x1, x2, x3, x4]  # output, intermediate

    def get_layers_features(self, x):
        # Just get the intermediate layers directly.

        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.avgpool(x4)
        x = x5.view(x.size(0), -1)
        # x = self.fc(x) # cooment this out for feature extraction for carla d4rl

        all_layers = [x0, x1, x2, x3, x4, x5, x]
        return all_layers


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:

        model_dict = model_zoo.load_url(model_urls['resnet18'])
        # remove the fc layers
        del model_dict['fc.weight']
        del model_dict['fc.bias']
        state = model.state_dict()
        state.update(model_dict)
        model.load_state_dict(state)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:

        model_dict = model_zoo.load_url(model_urls['resnet34'])
        # remove the fc layers
        del model_dict['fc.weight']
        del model_dict['fc.bias']
        state = model.state_dict()
        state.update(model_dict)
        model.load_state_dict(state)

    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_dict = model_zoo.load_url(model_urls['resnet50'])
        # remove the fc layers
        del model_dict['fc.weight']
        del model_dict['fc.bias']
        state = model.state_dict()
        state.update(model_dict)
        model.load_state_dict(state)

    return model



def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model