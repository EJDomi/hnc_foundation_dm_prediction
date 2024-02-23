from torch import nn
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.nn import global_mean_pool

class ClinicalGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, n_clinical, dropout=0.5):
        super(ClinicalGCN, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.classify = Linear(hidden_channels+n_clinical, num_classes)

        self.relu = ReLU()
        self.norm1 = BatchNorm(in_channels=hidden_channels)
        self.norm2 = BatchNorm(in_channels=hidden_channels)
        self.norm3 = BatchNorm(in_channels=hidden_channels)
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index, batch, clinical):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.norm1(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.norm2(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = self.norm3(x)
        x = self.dropout(x)
       
        x = global_mean_pool(x, batch)

        x = torch.cat((x, clinical), 1)

        x = self.dropout(x)
        x = self.classify(x)

        return x
