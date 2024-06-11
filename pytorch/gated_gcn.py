import torch
from torch import nn
from torch.nn import Linear, LeakyReLU, Dropout
from torch_geometric.nn import ResGatedGraphConv, BatchNorm
from torch_geometric.nn import global_mean_pool

class GatedGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, edge_dim=3, dropout=0.5):
        super(GatedGCN, self).__init__()

        self.conv1 = ResGatedGraphConv(in_channels, hidden_channels, edge_dim=edge_dim)
        self.conv2 = ResGatedGraphConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        self.conv3 = ResGatedGraphConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        self.classify = Linear(hidden_channels, num_classes)

        self.relu = LeakyReLU()
        self.norm1 = BatchNorm(in_channels=hidden_channels, allow_single_element=True)
        self.norm2 = BatchNorm(in_channels=hidden_channels, allow_single_element=True)
        self.norm3 = BatchNorm(in_channels=hidden_channels, allow_single_element=True)
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.dropout(x)
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.relu(x)
        x = self.norm1(x)
        #x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.norm2(x)
        #x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.norm3(x)
        #x = self.dropout(x)
       
        x = global_mean_pool(x, batch)

        x = self.dropout(x)
        x = self.classify(x)

        if len(x) == 1:
            x = x.squeeze().unsqueeze(0)
        else:
            x = x.squeeze()
        return x



class ClinicalGatedGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_clinical, edge_dim=1, dropout=0.5):
        super(ClinicalGatedGCN, self).__init__()

        self.conv1 = ResGatedGraphConv(in_channels, hidden_channels, edge_dim=edge_dim)
        self.conv2 = ResGatedGraphConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        self.conv3 = ResGatedGraphConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        self.classify = Linear(hidden_channels+num_clinical, num_classes)
        #self.linear = Linear(num_clinical, 64)
        #self.classify = Linear(hidden_channels, num_classes)

        self.relu = LeakyReLU()
        self.norm1 = BatchNorm(in_channels=hidden_channels, allow_single_element=True)
        self.norm2 = BatchNorm(in_channels=hidden_channels, allow_single_element=True)
        self.norm3 = BatchNorm(in_channels=hidden_channels, allow_single_element=True)
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch, clinical, radiomics=None):
        #if radiomics is not None:
        #    x = torch.cat((x,radiomics), 1)
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
       
        x = global_mean_pool(x, batch)

        x = torch.cat((x, clinical), 1)
        x = self.dropout(x)
        #x = self.linear(clinical)
        #x = self.relu(x)
        #x = self.dropout(x)
        x = self.classify(x)

        if len(x) == 1:
            x = x.squeeze().unsqueeze(0)
        else:
            x = x.squeeze()

         
        return x
