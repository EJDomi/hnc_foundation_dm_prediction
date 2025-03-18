import torch
from torch import nn
from torch.nn import LeakyReLU, Dropout
from torch_geometric.nn import ResGatedGraphConv, BatchNorm
from torch_geometric.nn import DeepGCNLayer, GENConv, GraphUNet
from torch_geometric.nn import GAT
from torch_geometric.nn import global_mean_pool



class EmptyNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_classes, edge_dim=None, dropout=0.5):
        super(EmptyNetwork, self).__init__()
        self.identity = nn.Identity()

    def forward(self, x, edge_index, batch, edge_attr=None):
        #x = global_mean_pool(x, batch)
        return self.identity(x)

class LinearNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_classes, edge_dim=3, dropout=0.0):
        super(LinearNet, self).__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, hidden_channels)
        self.linear3 = nn.Linear(hidden_channels, n_classes)
        self.relu = nn.LeakyReLU()
         

    def forward(self, x, edge_index=None, batch=None, edge_attr=None):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x) 
        x = self.relu(x)
        x = self.linear3(x) 
        x = self.relu(x)

        # the following is get the shape right so pytorch doesn't yell at you, 
        # in the off chance that the batch only has 1 entry
        if len(x) == 1:
            x = x.squeeze().unsqueeze(0)
        else:
            x = x.squeeze()
        return x


class GatedGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_classes, edge_dim=3, dropout=0.5):
        super(GatedGCN, self).__init__()

        self.conv1 = ResGatedGraphConv(in_channels, hidden_channels, edge_dim=edge_dim)
        self.conv2 = ResGatedGraphConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        self.conv3 = ResGatedGraphConv(hidden_channels, hidden_channels, edge_dim=edge_dim)

        self.relu = LeakyReLU()
        self.norm1 = BatchNorm(in_channels=hidden_channels, allow_single_element=True)
        self.norm2 = BatchNorm(in_channels=hidden_channels, allow_single_element=True)
        self.norm3 = BatchNorm(in_channels=hidden_channels, allow_single_element=True)
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.dropout(x)
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.relu(x)
        x = self.norm1(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.norm2(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.norm3(x)
        x = self.dropout(x)
       
        #x = global_mean_pool(x, batch)

        return x


class DeepGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_classes, edge_dim=1, dropout=0.5, num_layers=8):
        super(DeepGCN, self).__init__()

        self.layers = nn.ModuleList()

        conv = GENConv(in_channels, hidden_channels, aggr='softmax', t=0.1, learn_t=True, num_layers=2, norm='batch', edge_dim=edge_dim)
        norm = BatchNorm(hidden_channels)
        act = LeakyReLU(inplace=True)
        layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=dropout)
        self.layers.append(layer)

        for i in range(1, num_layers): 
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax', t=0.1, learn_t=True, num_layers=2, norm='batch', edge_dim=edge_dim)
            norm = BatchNorm(hidden_channels)
            act = LeakyReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=dropout)
            self.layers.append(layer)
           
        self.dropout = Dropout(dropout)


    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x = self.dropout(x)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = self.dropout(x)
 
        x = global_mean_pool(x, batch)

        return x


class myGraphUNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_classes, edge_dim=3, dropout=0.5, num_layers=8):
        super(myGraphUNet, self).__init__()

        self.graphunet = GraphUNet(in_channels, hidden_channels, out_channels=hidden_channels, depth=4)

        self.dropout = Dropout(dropout)


    def forward(self, x, edge_index, batch, clinical, edge_attr=None):
        x = self.graphunet(x, edge_index, batch)

        x = global_mean_pool(x, batch)

        return x


class myGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_classes, edge_dim=3, dropout=0.5, num_layers=3):
        super(myGAT, self).__init__()

        self.gat = GAT(in_channels, hidden_channels, out_channels=hidden_channels, dropout=dropout, norm='BatchNorm', num_layers=3)

        self.dropout = Dropout(dropout)


    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.gat(x, edge_index, batch)

        #x = global_mean_pool(x, batch)

        return x
