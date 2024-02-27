import torch
from torch import nn
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import GraphUNet, BatchNorm
from torch_geometric.nn import global_mean_pool

class myGraphUNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_clinical, dropout=0.5):
        super(myGraphUNet, self).__init__()

        self.graphunet = GraphUNet(in_channels, hidden_channels, out_channels=hidden_channels, depth=4)
        self.classify = Linear(hidden_channels+num_clinical, num_classes)

        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index, batch, clinical):
        x = self.graphunet(x, edge_index, batch)

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
