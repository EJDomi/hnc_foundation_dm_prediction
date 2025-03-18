import torch
from torch import nn
from torch.nn import LeakyReLU, Dropout

from hnc_project.pytorch.resnet_lightning import * 
from hnc_project.pytorch.sgc_cnn import SGC_CNN 
from hnc_project.pytorch.densenet import DenseNet3d 
from hnc_project.pytorch.net_swin import SwinTransformer
from hnc_project.pytorch.resnet_spottune import SpotTune
#from hnc_project.pytorch.net_vit import ViT
from monai.networks.nets.vit import ViT



def resnet18(n_classes=1, in_channels=3, dropout=0.0, blocks=BasicBlock):
    return ResNet(blocks, [2,2,2,2], in_channels=in_channels, n_classes=n_classes, dropout=dropout)

def resnet34(n_classes=1, in_channels=3, dropout=0.0, blocks=BasicBlock):
    return ResNet(blocks, [3,4,6,3], in_channels=in_channels, n_classes=n_classes, dropout=dropout)

def resnet50(n_classes=1, in_channels=3, dropout=0.0, blocks=Bottleneck, config=None):
    return ResNet(blocks, [3,4,6,3], in_channels=in_channels, n_classes=n_classes, dropout=dropout)

def resnet101(n_classes=1, in_channels=3, dropout=0.0, blocks=Bottleneck):
    return ResNet(blocks, [3,4,23,3], in_channels=in_channels, n_classes=n_classes, dropout=dropout)

def resnet152(n_classes=1, in_channels=3, dropout=0.0, blocks=Bottleneck):
    return ResNet(blocks, [3,8,36,3], in_channels=in_channels, n_classes=n_classes, dropout=dropout)
def resnet200(n_classes=1, in_channels=3, dropout=0.0, blocks=Bottleneck):
    return ResNet(blocks, [3,24,36,3], in_channels=in_channels, n_classes=n_classes, dropout=dropout)

def densenet3d(n_classes=1, in_channels=64, dropout = 0.0):
    return DenseNet3d(num_classes=n_classes, in_channels=in_channels, dropout_p=dropout)

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



def swin(n_classes=1, in_channels=3, dropout = 0.0):
    return SwinTransformer(patch_size=2,
            in_chans=in_channels, 
            embed_dim=96, 
            depths=(2,2,4,2),
            num_heads=(4,4,8,8),
            window_size=(7,7,7),
            mlp_ratio=4,
            qkv_bias=False,
            drop_rate=dropout,
            drop_path_rate=0.3,
            ape=False, spe=False, rpe=True, patch_norm=True, use_checkpoint=False,
            out_indices=(0,1,2,3),
            pat_merg_rf=4)

def vit(n_classes=1, in_channels=3, dropout=0.0):
    return ViT(
        in_channels=in_channels,
        img_size = (80,80,80),
        patch_size = 16,
        hidden_size = 768,
        mlp_dim = 3072,
        num_layers = 10,
        num_heads = 12,
        pos_embed = "conv",
        classification = True,
        num_classes = n_classes,
        dropout_rate = dropout,
        spatial_dims = 3,
        )


class EmptyNetwork(nn.Module):
    def __init__(self, n_classes, in_channels=2, dropout=0.5):
        super(EmptyNetwork, self).__init__()
        self.identity = nn.Identity()

    def forward(self, x):
        #x = global_mean_pool(x, batch)
        return self.identity(x)


class LinearNet(nn.Module):
    def __init__(self, n_classes=1, in_channels=1316, dropout=0.0):
        super(LinearNet, self).__init__()
        self.linear1 = nn.Linear(in_channels, 2)
        self.linear2 = nn.Linear(2, 2)
        self.linear3 = nn.Linear(2, n_classes)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, clinical=None):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        if clinical is not None:
            #clinical = torch.unique(clinical, dim=0)
            x = torch.cat((x, clinical), 1)
        else:
            x = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x) 
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x) 
        x = self.relu(x)
        x = self.dropout(x)

        # the following is get the shape right so pytorch doesn't yell at you, 
        # in the off chance that the batch only has 1 entry
        if len(x) == 1:
            x = x.squeeze().unsqueeze(0)
        else:
            x = x.squeeze()
        return x
 
class LNCNN(nn.Module):
    def __init__(self, n_classes, in_channels, dropout):
        super(LNCNN, self).__init__()

        self.cn1 = nn.Conv3d(in_channels, 64, kernel_size=(5,5,5), stride=(1,1,1), padding='valid', bias=False)
        self.cn2 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding='valid', bias=False)
        self.cn3 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding='valid', bias=False)
        self.cn4 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding='valid', bias=False)
        self.cn5 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding='valid', bias=False)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(3,3,3), stride=2)
        self.cn6 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding='same', bias=False)
        self.cn7 = nn.Conv3d(64, 64, kernel_size=(2,2,2), stride=(1,1,1), padding='same', bias=False)
        self.cn8 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding='same', bias=False)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(3,3,3), stride=2)
        self.cn9 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding='same', bias=False)
        self.cn10 = nn.Conv3d(64, 64, kernel_size=(3,3,1), stride=(1,1,1), padding='same', bias=False)
        self.cn11 = nn.Conv3d(64, 64, kernel_size=(3,3,1), stride=(1,1,1), padding='same', bias=False)
        self.cn12 = nn.Conv3d(64, 32, kernel_size=(3,3,1), stride=(1,1,1), padding='same', bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(64)
        self.bn4 = nn.BatchNorm3d(64)
        self.bn5 = nn.BatchNorm3d(64)
        self.bn6 = nn.BatchNorm3d(64)
        self.bn7 = nn.BatchNorm3d(64)
        self.bn8 = nn.BatchNorm3d(64)
        self.bn9 = nn.BatchNorm3d(64)
        self.bn10 = nn.BatchNorm3d(64)
        self.bn11 = nn.BatchNorm3d(64)
        self.bn12 = nn.BatchNorm3d(32)

        self.dropout = nn.Dropout(dropout)
        self.dropout05 = nn.Dropout(0.05)
        self.flatten = nn.Flatten()
        #self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        # 102752 for 80x80x80
        self.linear = nn.Linear(102752, n_classes)
        #self.classify = nn.Linear(256, n_classes)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.cn1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout05(x)
        x = self.cn2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout05(x)
        x = self.cn3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout05(x)
        x = self.cn4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout05(x)
        x = self.cn5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.dropout05(x)
        x = self.maxpool1(x)
        x = self.cn6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.dropout05(x)
        x = self.cn7(x)
        x = self.bn7(x)
        x = self.relu(x)
        x = self.dropout05(x)
        x = self.cn8(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.dropout05(x)
        x = self.maxpool2(x)
        x = self.cn9(x)
        x = self.bn9(x)
        x = self.relu(x)
        x = self.dropout05(x)
        x = self.cn10(x)
        x = self.bn10(x)
        x = self.relu(x)
        x = self.dropout05(x)
        x = self.cn11(x)
        x = self.bn11(x)
        x = self.relu(x)
        x = self.dropout05(x)
        x = self.cn12(x)
        x = self.bn12(x)
        x = self.relu(x)
        x = self.dropout05(x)
        #x = self.avgpool(x)
        x = self.flatten(x)
         
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)

        #x = self.classify(x)

        return x.squeeze()
