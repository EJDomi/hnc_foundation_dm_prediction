import torch
from torch import nn
from torch.nn import Linear, ReLU, Dropout, Identity
from torch_geometric.nn import GCNConv, BatchNorm, ResGatedGraphConv
from torch_geometric.nn import global_mean_pool
from hnc_project.pytorch.resnet import resnet50
from hnc_project.pytorch.transfer_layer_translation_cfg import layer_loop, layer_loop_downsample
 
class ResNetGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.5):
        super(ResNetGCN, self).__init__()

       
        self.resnet = resnet50(num_classes=num_classes, in_channels=in_channels, dropout=dropout) 
        self.resnet.classify = Identity()

        self.conv1 = GCNConv(self.resnet.in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.classify = Linear(hidden_channels, num_classes)

        self.relu = ReLU()
        self.norm1 = BatchNorm(in_channels=hidden_channels)
        self.norm2 = BatchNorm(in_channels=hidden_channels)
        self.norm3 = BatchNorm(in_channels=hidden_channels)
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index, batch):
        try:
            x = self.resnet(x)
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

            x = self.dropout(x)
            x = self.classify(x)
        except:
            print(f"x: {x}")
            print(f"edge_index: {edge_index}")

        return x.squeeze()

 
class GatedResNetGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, edge_dim=3, dropout=0.5):
        super(GatedResNetGCN, self).__init__()

       
        self.resnet = resnet50(num_classes=num_classes, in_channels=in_channels, dropout=dropout) 
        self.resnet.classify = Identity()

        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #initial_state = torch.load('./models/resnet_50.pth', map_location=device)['state_dict']
        #fixed_state = {}
        #for k, v in initial_state.items():
        #    if 'layer' in k:
        #        mod_name = k.replace('module', 'blocks')
        #    else:
        #        mod_name = k.replace('module.', '')
        #    for name, new in layer_loop.items():
        #        if name in mod_name:
        #            mod_name = mod_name.replace(name, new)
        #    for name, new in layer_loop_downsample.items():
        #        if name in mod_name:
        #            mod_name = mod_name.replace(name, new)
        #    fixed_state[mod_name] = v

        #if in_channels > 1:
        #    fixed_state['conv1.weight'] = fixed_state['conv1.weight'].repeat(1,in_channels,1,1,1)/in_channels  

        #self.resnet.load_state_dict(fixed_state, strict=False)
        #for name, p in self.resnet.named_parameters():
        #    p.requires_grad = False
        
        self.conv1 = ResGatedGraphConv(self.resnet.in_channels, hidden_channels, edge_dim=edge_dim)
        self.conv2 = ResGatedGraphConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        self.conv3 = ResGatedGraphConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        self.classify = Linear(hidden_channels, num_classes)

        self.relu = ReLU()
        self.norm1 = BatchNorm(in_channels=hidden_channels)
        self.norm2 = BatchNorm(in_channels=hidden_channels)
        self.norm3 = BatchNorm(in_channels=hidden_channels)
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.resnet(x)
        x = self.dropout(x)
        x = self.conv1(x, edge_index, edge_attr)
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

        x = self.dropout(x)
        x = self.classify(x)

        return x.squeeze()
