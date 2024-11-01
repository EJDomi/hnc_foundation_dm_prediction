import torch
from torch import nn
import math

class SGC_CNN(nn.Module):
    def __init__(self, in_channels, dropout):
        super(SGC_CNN, self).__init__()
 

        self.cn1 = nn.Sequential(
            nn.LazyConv3d(16, (3,3,3), (1,1,1), 1),
            nn.BatchNorm3d(),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, (3,3,3), (1,1,1), 1),
            nn.BatchNorm3d(),
            nn.ReLU(inplace=True),
        )
        self.cn1_pool = nn.MaxPool3d(kernel_size=(2,2,2))

        self.cn2 = nn.Sequential(
            nn.Conv3d(16, 32, (3,3,3), (1,1,1), 1),
            nn.BatchNorm3d(),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, (3,3,3), (1,1,1), 1),
            nn.BatchNorm3d(),
            nn.ReLU(inplace=True),
        )
        self.cn2_pool = nn.MaxPool3d(kernel_size=(2,2,2))

        self.cn3 = nn.Sequential(
            nn.Conv3d(32, 64, (3,3,3), (1,1,1), 1),
            nn.BatchNorm3d(),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, (3,3,3), (1,1,1), 1),
            nn.BatchNorm3d(),
            nn.ReLU(inplace=True),
        )
        self.cn3_pool = nn.MaxPool3d(kernel_size=(2,2,2))

        self.cn4 = nn.Sequential(
            nn.Conv3d(64, 128, (3,3,3), (1,1,1), 1),
            nn.BatchNorm3d(),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, (3,3,3), (1,1,1), 1),
            nn.BatchNorm3d(),
            nn.ReLU(inplace=True),
        )
        self.cn4_pool = nn.MaxPool3d(kernel_size=(2,2,2))

        self.cn5 = nn.Sequential(
            nn.Conv3d(128, 256, (3,3,3), (1,1,1), 1),
            nn.BatchNorm3d(),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, (3,3,3), (1,1,1), 1),
            nn.BatchNorm3d(),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(256, 128, (2,2,2), (2,2,2), 1),
            nn.BatchNorm3d(),
            nn.ReLU(inplace=True),
        )

        self.d1 = nn.Sequential(
            nn.Conv3d(128, 128, (3,3,3), (1,1,1), 1),
            nn.BatchNorm3d(),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, (3,3,3), (1,1,1), 1),
            nn.BatchNorm3d(),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, (2,2,2), (2,2,2), 1),
            nn.BatchNorm3d(),
            nn.ReLU(inplace=True),
        )

        self.d2 = nn.Sequential(
            nn.Conv3d(64, 64, (3,3,3), (1,1,1), 1),
            nn.BatchNorm3d(),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, (3,3,3), (1,1,1), 1),
            nn.BatchNorm3d(),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, (2,2,2), (2,2,2), 1),
            nn.BatchNorm3d(),
            nn.ReLU(inplace=True),
        )
        
        self.d3 = nn.Sequential(
            nn.Conv3d(32, 32, (3,3,3), (1,1,1), 1),
            nn.BatchNorm3d(),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, (3,3,3), (1,1,1), 1),
            nn.BatchNorm3d(),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, (2,2,2), (2,2,2), 1),
            nn.BatchNorm3d(),
            nn.ReLU(inplace=True),
        )
        
        self.d4 = nn.Sequential(
            nn.Conv3d(16, 16, (3,3,3), (1,1,1), 1),
            nn.BatchNorm3d(),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, (3,3,3), (1,1,1), 1),
            nn.BatchNorm3d(),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, (3,3,3), (1,1,1), 1),
            nn.BatchNorm3d(),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, (3,3,3), (1,1,1), 1),
            nn.ReLU(inplace=True),
        )

        self.threshold = nn.Lambda(self.threshold_seg) 
 
 
        self.cnn = []
     
        self.cnn.append(self.cnn_bn_relu((in_channels, 64), (5,5,5), (1,1,1), 0))

        for b in range(4):
            self.cnn.append(self.cnn_bn_relu((64,64), (3,3,3), (1,1,1), 0))

        self.cnn.append(nn.MaxPool3d(kernel_size=(2,2,2), stride=2))       

        self.cnn.append(self.cnn_bn_relu((64,64), (3,3,3), (1,1,1), 0)) 
        self.cnn.append(self.cnn_bn_relu((64,64), (2,2,2), (1,1,1), 0)) 
        self.cnn.append(self.cnn_bn_relu((64,64), (3,3,3), (1,1,1), 0)) 

        self.cnn.append(nn.MaxPool3d(kernel_size=(3,3,3), stride=2))       

        self.cnn.append(self.cnn_bn_relu((64,64), (3,3,3), (1,1,1), 1))
        self.cnn.append(self.cnn_bn_relu((64,64), (3,3,1), (1,1,1), 1))
        self.cnn.append(self.cnn_bn_relu((64,64), (3,3,1), (1,1,1), 1))
        self.cnn.append(self.cnn_bn_relu((64,32), (3,3,1), (1,1,1), 1))

        self.cnn.append(nn.Flatten())
        self.cnn.append(nn.LazyLinear(out_features=256))

        self.cnn = nn.Sequential(*self.cnn)

        #self.classify = nn.Linear(256, 2)
 

    def cnn_bn_relu(channels=(64,64), kernel=(3,3,3), stride=(1,1,1), padding=0):
        layer = []
        layer.append(nn.Conv3d(channels[0], channels[1], kernel_size=kernel, stride=stride, padding=padding, bias=False))
        layer.append(nn.BatchNorm3d())
        layer.append(nn.ReLU(inplace=True))
        layer.append(nn.Dropout(0.05))

        return layer

       
    def threshold_seg(self, x):
        return torch.where(torch.equal(x[0], torch.zeros(x[0].size()), torch.zeros(x[0].size()), x[1]))


    def forward(self, x, clinical=None):

        xi = x

        x1 = self.cn1(x)
        x1_pool = self.cn1_pool(x1)
 
        x2 = self.cn2(x1_pool)
        x2_pool = self.cn2_pool(x2)
        
        x3 = self.cn3(x2_pool)
        x3_pool = self.cn3_pool(x3)

        x4 = self.cn4(x3_pool)
        x4_pool = self.cn4_pool(x4)

        x5 = self.cn5(x4_pool)

        xd = torch.cat((x5, x4), 1)
        xd = self.d1(xd)

        xd = torch.cat((xd, x3), 1)  
        xd = self.d2(xd)

        xd = torch.cat((xd, x2), 1)  
        xd = self.d3(xd)

        xd = torch.cat((xd, x1), 1)  
        xd = self.d4(xd)

        #xd is the activation map, what we need to compare the seg to
        xout = self.threshold((x, xd))

        xout = torch.multiply(xout, xi) 
        xout = self.cnn(xout)

        #if clinical is not None:
        #    xout = torch.cat((xout, clinical), 1)

        #xout = self.classify(xout)

        return xout, xd

    
       
