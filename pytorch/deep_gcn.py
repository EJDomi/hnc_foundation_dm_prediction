import torch
from torch import nn
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import DeepGCNLayer, GENConv, BatchNorm
from torch_geometric.nn import global_mean_pool


class DeepGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, num_classes, num_clinical, edge_dim=1, dropout=0.5):
        super(DeepGCN, self).__init__()

        self.layers = nn.ModuleList()

        conv = GENConv(in_channels, hidden_channels, aggr='softmax', t=0.1, learn_t=True, num_layers=2, norm='batch', edge_dim=edge_dim)
        norm = BatchNorm(hidden_channels)
        act = ReLU(inplace=True)
        layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=dropout)
        self.layers.append(layer)

        for i in range(1, num_layers): 
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax', t=0.1, learn_t=True, num_layers=2, norm='batch', edge_dim=edge_dim)
            norm = BatchNorm(hidden_channels)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=dropout)
            self.layers.append(layer)
           

        self.classify = Linear(hidden_channels+num_clinical, num_classes)
        #self.classify = Linear(hidden_channels, num_classes)

        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch, clinical):
        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x = self.dropout(x)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = self.dropout(x)
 
        x = global_mean_pool(x, batch)

        x = torch.cat((x, clinical), 1)
        x = self.dropout(x)
        x = self.classify(x)

        if len(x) == 1:
            x = x.squeeze().unsqueeze(0)
        else:
            x = x.squeeze()

        return x


class AltDeepGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, num_classes, num_clinical, edge_dim=1, dropout=0.5):
        super(AltDeepGCN, self).__init__()

        self.layers = nn.ModuleList()

        conv = GENConv(in_channels, hidden_channels, aggr='softmax', t=0.1, learn_t=True, num_layers=2, norm='batch', edge_dim=edge_dim)
        norm = BatchNorm(hidden_channels)
        act = ReLU(inplace=True)
        layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=dropout)
        self.layers.append(layer)

        for i in range(1, num_layers): 
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax', t=0.1, learn_t=True, num_layers=2, norm='batch', edge_dim=edge_dim)
            norm = BatchNorm(hidden_channels)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=dropout)
            self.layers.append(layer)
           

        self.classify = Linear(hidden_channels+num_clinical, num_classes)

        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch, clinical, radiomics):

        x = torch.cat((x, radiomics), 1)
        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = self.dropout(x)
 
        x = global_mean_pool(x, batch)

        x = torch.cat((x, clinical), 1)
        x = self.dropout(x)
        x = self.classify(x)

        if len(x) == 1:
            x = x.squeeze().unsqueeze(0)
        else:
            x = x.squeeze()

        return x
