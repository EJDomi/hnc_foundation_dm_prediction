import torch
from torch import nn
from torch.nn import LeakyReLU, Dropout

from hnc_project.pytorch.resnet_lightning import * 
from hnc_project.pytorch.sgc_cnn import SGC_CNN 
from hnc_project.pytorch.densenet import DenseNet3d 
from hnc_project.pytorch.net_swin import SwinTransformer




def resnet18(n_classes=1, in_channels=3, dropout=0.0, blocks=BasicBlock):
    return ResNet(blocks, [2,2,2,2], in_channels=in_channels, n_classes=n_classes, dropout=dropout)

def resnet34(n_classes=1, in_channels=3, dropout=0.0, blocks=BasicBlock):
    return ResNet(blocks, [3,4,6,3], in_channels=in_channels, n_classes=n_classes, dropout=dropout)

def resnet50(n_classes=1, in_channels=3, dropout=0.0, blocks=Bottleneck):
    return ResNet(blocks, [3,4,6,3], in_channels=in_channels, n_classes=n_classes, dropout=dropout)

def resnet101(n_classes=1, in_channels=3, dropout=0.0, blocks=Bottleneck):
    return ResNet(blocks, [3,4,23,3], in_channels=in_channels, n_classes=n_classes, dropout=dropout)

def resnet152(n_classes=1, in_channels=3, dropout=0.0, blocks=Bottleneck):
    return ResNet(blocks, [3,8,36,3], in_channels=in_channels, n_classes=n_classes, dropout=dropout)
def resnet200(n_classes=1, in_channels=3, dropout=0.0, blocks=Bottleneck):
    return ResNet(blocks, [3,24,36,3], in_channels=in_channels, n_classes=n_classes, dropout=dropout)

def densenet3d(n_classes=1, in_channels=64, dropout = 0.0):
    return DenseNet3d(num_classes=n_classes, in_channels=in_channels, dropout_p=dropout)

def swin(n_classes=1, in_channels=64, dropout = 0.0):
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

class LNCNN(nn.Module):
    def __init__(self, n_classes, in_channels, dropout):
        super(LNCNN, self).__init__()

        self.cn1 = nn.Conv3d(in_channels, 64, kernel_size=(5,5,5), stride=(1,1,1), padding=0, bias=False)
        self.cn2 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=0, bias=False)
        self.cn3 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=0, bias=False)
        self.cn4 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=0, bias=False)
        self.cn5 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=0, bias=False)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(3,3,3), stride=2)
        self.cn6 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=0, bias=False)
        self.cn7 = nn.Conv3d(64, 64, kernel_size=(2,2,2), stride=(1,1,1), padding=0, bias=False)
        self.cn8 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), bias=False)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(3,3,3), stride=2)
        self.cn9 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=1, bias=False)
        self.cn10 = nn.Conv3d(64, 64, kernel_size=(3,3,1), stride=(1,1,1), padding=1, bias=False)
        self.cn11 = nn.Conv3d(64, 64, kernel_size=(3,3,1), stride=(1,1,1), padding=1, bias=False)
        self.cn12 = nn.Conv3d(64, 32, kernel_size=(3,3,1), stride=(1,1,1), padding=1, bias=False)

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
        self.flatten = nn.Flatten()
        #self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        # 102752 for 80x80x80
        self.linear = nn.Linear(102752, 256)
        self.classify = nn.Linear(256, n_classes)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.cn1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.cn2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.cn3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.cn4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.cn5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.cn6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.cn7(x)
        x = self.bn7(x)
        x = self.relu(x)
        x = self.cn8(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.cn9(x)
        x = self.bn9(x)
        x = self.relu(x)
        x = self.cn10(x)
        x = self.bn10(x)
        x = self.relu(x)
        x = self.cn11(x)
        x = self.bn11(x)
        x = self.relu(x)
        x = self.cn12(x)
        x = self.bn12(x)
        x = self.relu(x)
        #x = self.avgpool(x)
        x = self.flatten(x)
         
        x = self.linear(x)
        x = self.dropout(x)

        x = self.classify(x)

        return x.squeeze()
