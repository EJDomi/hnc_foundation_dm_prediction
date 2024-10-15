#!/usr/bin/env python
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.dilation = dilation
        if stride > 1 or dilation > 1:
            self.downsample = nn.Conv3d(in_channels, channels, kernel_size=1, stride=stride, bias=False)
            self.bn0 = nn.BatchNorm3d(channels)
            self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=(3,3,3), stride=stride, dilation=dilation, padding=dilation, bias=False)
        else:
            self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=(3,3,3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=(3,3,3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.stride > 1 or self.dilation > 1:
            residual = self.downsample(residual)
            residual = self.bn0(residual)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.dilation = dilation
        if stride > 1:
            self.downsample = nn.Conv3d(in_channels, channels*self.expansion, kernel_size=1, stride=stride, bias=False)
            self.bn0 = nn.BatchNorm3d(channels*self.expansion)
        elif in_channels != channels*self.expansion:
            self.downsample = nn.Conv3d(in_channels, channels*self.expansion, kernel_size=1, stride=1, bias=False)
            self.bn0 = nn.BatchNorm3d(channels*self.expansion)
        else:
            self.downsample = None
        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=(1,1,1), stride=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU()
        if stride > 1 or dilation > 1:
            self.conv2 = nn.Conv3d(channels, channels, kernel_size=(3,3,3), stride=stride, padding=dilation, dilation=dilation, bias=False)
        else:
            self.conv2 = nn.Conv3d(channels, channels, kernel_size=(3,3,3), stride=1, padding=1, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.conv3 = nn.Conv3d(channels, channels*self.expansion, kernel_size=(1,1,1), stride=1, bias=False)
        self.bn3 = nn.BatchNorm3d(channels*self.expansion)

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

        if self.downsample:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out



class MainResNet(nn.Module):
    def __init__(self, block, layers, in_channels=3, num_classes=1, dropout=0.0):
        super(MainResNet, self).__init__()
        if block == BasicBlock:
            self.expansion = 1
        if block == Bottleneck:
            self.expansion = 4 
        self.factor = 2
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(7,7,7), stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(3,3,3), stride=2, dilation=1, padding=1)

        self.in_channels = 64
        #self.strides = [1, 2, 2, 2]
        self.strides = [1, 2, 1, 1]
        self.dilations = [1, 1, 2, 4]
        self.channels = [64, 128, 256, 512]
        #self.nblocks = [3, 4, 6, 3] -> layers
        self.layers = layers

        self.blocks = []
        self.parallel_blocks = []

        for idx, (ch, n_blocks, stride, dilation) in enumerate(zip(self.channels, self.layers, self.strides, self.dilations)):
            blocks = self._make_layer(block, ch, n_blocks, stride=stride, dilation=dilation)
            self.blocks.append(nn.ModuleList(blocks))
        self.blocks = nn.ModuleList(self.blocks)
    
        self.in_channels = 64
        for idx, (ch, n_blocks, stride, dilation) in enumerate(zip(self.channels, self.layers, self.strides, self.dilations)):
            blocks = self._make_layer(block, ch, n_blocks, stride=stride, dilation=dilation)
            self.parallel_blocks.append(nn.ModuleList(blocks))
        self.parallel_blocks = nn.ModuleList(self.parallel_blocks)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.classify = nn.Linear(self.channels[-1]*self.expansion, num_classes)
        self.dropout = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def _make_layer(self, block, channels, n_blocks, stride=1, dilation=1):

        layers = []
        layers.append(block(self.in_channels, channels, stride=stride, dilation=dilation))
        self.in_channels = channels * block.expansion
        for i in range(1, n_blocks):
            layers.append(block(self.in_channels, channels))

        return layers



    def forward(self, x, policy=None):
        t = 0

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)


        if policy is not None:
            for segment, n_blocks in enumerate(self.layers):
                for b in range(n_blocks):
                    action = policy[:,t].contiguous()
                    action_mask = action.float().view(-1,1,1,1,1)

                    output = self.blocks[segment][b](x)
                    output_ = self.parallel_blocks[segment][b](x)

                    f1 = output
                    f2 = output_
                    x = f1*(1-action_mask) + f2*action_mask
                    x = self.dropout(x)
                    t += 1

        else:
            for segment, n_blocks in enumerate(self.layers):
                for b in range(n_blocks):
                    output = self.blocks[segment][b](x)
                    x = output
                    x = self.dropout(x)
                    t += 1

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classify(x)

        return x


class Agent(nn.Module):
    def __init__(self, block, layers, in_channels=3, num_classes=1, dropout=0.0):
        super(Agent, self).__init__()
        self.expansion = 1
        self.factor = 1
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(7,7,7), stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(3,3,3), stride=2, dilation=1, padding=1)

        self.in_channels = 64
        #self.strides = [1, 2, 2, 2]
        self.strides = [1, 2, 1, 1]
        self.dilations = [1, 1, 2, 4]
        self.channels = [64, 128, 256, 512]
        #self.nblocks = [3, 4, 6, 3] -> layers
        self.layers = layers

        self.blocks = []
        self.parallel_blocks = []

        for idx, (ch, n_blocks, stride, dilation) in enumerate(zip(self.channels, self.layers, self.strides, self.dilations)):
            blocks = self._make_layer(block, ch, n_blocks, stride=stride, dilation=dilation)
            self.blocks.append(nn.ModuleList(blocks))
        self.blocks = nn.ModuleList(self.blocks)
    
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.classify = nn.Linear(self.channels[-1]*self.expansion, num_classes)
        self.dropout = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channels, n_blocks, stride=1, dilation=1):

        layers = []
        layers.append(block(self.in_channels, channels, stride=stride, dilation=dilation))
        self.in_channels = channels * block.expansion
        for i in range(1, n_blocks):
            layers.append(block(self.in_channels, channels))

        return layers



    def forward(self, x, policy=None):
        t = 0

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)


        if policy is not None:
            for segment, n_blocks in enumerate(self.layers):
                for b in range(n_blocks):
                    output = self.blocks[segment][b](x)
                    x = output
                    x = self.dropout(x)
                    t += 1

        else:
            for segment, n_blocks in enumerate(self.layers):
                for b in range(n_blocks):
                    output = self.blocks[segment][b](x)
                    x = output
                    x = self.dropout(x)
                    t += 1

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classify(x)

        return x





class SpotTune(nn.Module):
    def __init__(self, main='main18', in_channels=2, num_classes=64, dropout=0.0):
        super(SpotTune, self).__init__()

        weight_map = {
            'main18': 'resnet_18.pth',
            'main34': 'resnet_34.pth',
            'main50': 'resnet_50.pth',
            'main101': 'resnet_101.pth',
            'main152': 'resnet_152.pth',
            'main200': 'resnet_200.pth',
        }

        self.resnet = globals()[main](n_classes=num_classes, in_channels=in_channels, dropout=dropout)
        self.agent = Agent(BasicBlock, [1,1,1,1], in_channels=in_channels, num_classes=(sum(self.resnet.layers)*2), dropout=dropout)

        initial_state = torch.load(os.path.join('./weights', weight_map[main]))['state_dict']
        fixed_state = OrderedDict()
        for k, v in initial_state.items():
            if 'layer' in k:
                mod_name = k.replace('module', 'blocks')
            else:
                mod_name = k.replace('module.', '')
            for name, new in layer_loop.items():
                if name in mod_name:
                    mod_name = mod_name.replace(name, new)
            for name, new in layer_loop_downsample.items():
                if name in mod_name:
                    mod_name = mod_name.replace(name, new)
            fixed_state[mod_name] = v

        fixed_state_v2 = OrderedDict()
        for k, v in fixed_state.items():
            fixed_state_v2[k] = v
            fixed_state_v2[k.replace('blocks', 'parallel_blocks')] = v

        if self.n_channels > 1:
            fixed_state_v2['conv1.weight'] = fixed_state['conv1.weight'].repeat(1,self.n_channels,1,1,1)/self.n_channels

        self.resnet.load_state_dict(fixed_state_v2, strict=False)
        self.freeze_layers(ignore=['classify', 'parallel_blocks'])
        for name, p in self.resnet.named_parameters():
            if any([i in name for i in ['classify', 'parallel_blocks']]):
                p.requires_grad=True
            else:
                p.requires_grad=False

         

    def forward(self, x):
        probs = self.agent(x)
        action = self.gumbel_softmax(probs.view(probs.size(0), -1, 2), 5)
        policy = action[:,:,1]

        x = self.resnet(x, policy)

        return x



    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.cuda.FloatTensor(shape).uniform_()
        #U = torch.FloatTensor(shape).uniform_()
        return -Variable(torch.log(-torch.log(U + eps) + eps))


    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)


    def gumbel_softmax(self, logits, temperature = 5):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y


def main18(n_classes=1, in_channels=3, dropout=0.0, blocks=BasicBlock):
    return ResNet(blocks, [2,2,2,2], in_channels=in_channels, num_classes=n_classes, dropout=dropout)

def main34(n_classes=1, in_channels=3, dropout=0.0, blocks=BasicBlock):
    return ResNet(blocks, [3,4,6,3], in_channels=in_channels, num_classes=n_classes, dropout=dropout)

def main50(n_classes=1, in_channels=3, dropout=0.0, blocks=Bottleneck):
    return ResNet(blocks, [3,4,6,3], in_channels=in_channels, num_classes=n_classes, dropout=dropout)

def main101(n_classes=1, in_channels=3, dropout=0.0, blocks=Bottleneck):
    return ResNet(blocks, [3,4,23,3], in_channels=in_channels, num_classes=n_classes, dropout=dropout)

def main152(n_classes=1, in_channels=3, dropout=0.0, blocks=Bottleneck):
    return ResNet(blocks, [3,8,36,3], in_channels=in_channels, num_classes=n_classes, dropout=dropout)

def main200(n_classes=1, in_channels=3, dropout=0.0, blocks=Bottleneck):
    return ResNet(blocks, [3,24,36,3], in_channels=in_channels, num_classes=n_classes, dropout=dropout)


def spottune18(n_classes=1, in_channels=3, dropout=0.0, blocks=BasicBlock):
    return SpotTune(main='main18', in_channels=in_channels, num_classes=n_classes, dropout=dropout)

def spottune34(n_classes=1, in_channels=3, dropout=0.0, blocks=BasicBlock):
    return SpotTune(main='main34', in_channels=in_channels, num_classes=n_classes, dropout=dropout)

def spottune50(n_classes=1, in_channels=3, dropout=0.0, blocks=Bottleneck):
    return SpotTune(main='main50', in_channels=in_channels, num_classes=n_classes, dropout=dropout)

def spottune101(n_classes=1, in_channels=3, dropout=0.0, blocks=Bottleneck):
    return SpotTune(main='main101', in_channels=in_channels, num_classes=n_classes, dropout=dropout)

def spottune152(n_classes=1, in_channels=3, dropout=0.0, blocks=Bottleneck):
    return SpotTune(main='main152', in_channels=in_channels, num_classes=n_classes, dropout=dropout)

def spottune200(n_classes=1, in_channels=3, dropout=0.0, blocks=Bottleneck):
    return SpotTune(main='main200', in_channels=in_channels, num_classes=n_classes, dropout=dropout)

weight_map = {
    'main18': 'resnet_18.pth',
    'main34': 'resnet_34.pth',
    'main50': 'resnet_50.pth',
    'main101': 'resnet_101.pth',
    'main152': 'resnet_152.pth',
    'main200': 'resnet_200.pth',
}
