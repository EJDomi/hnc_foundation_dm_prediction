import torch
from torch import nn
import torch.nn.functional as F

import torchvision
import torchmetrics

import lightning as L

import LN_project_repo.pytorch.extractor_networks as en
import LN_project_repo.pytorch.gnn_networks as graphs



class Classify(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.classify = nn.LazyLinear(n_classes)


    def forward(self, x, clinical=None):
        if clinical is not None:
            x = torch.cat((x, clinical), 1)
            x = self.classify(x) 
        else:
            x = self.classify(x)

        if len(x) == 1:
            x = x.squeeze().unsqueeze(0)
        else:
            x = x.squeeze()
        return x


class CNN_GNN(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.learning_rate = self.config['learning_rate']
        self.extractor = getattr(en, self.config['extractor_name'])(n_classes=self.config['n_extracted_channels'], in_channels=self.config['n_in_channels'], dropout=self.config['dropout'])
        self.gnn = getattr(graphs, self.config['model_name'])(in_channels=self.config['n_extracted_channels'], hidden_channels=self.config['n_hidden_channels'], n_classes=self.config['n_hidden_channels'], edge_dim=self.config['edge_dim'], dropout=self.config['dropout'])
 
        self.classify = Classify(n_classes=self.config['n_classes'])

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.auc_fn = torchmetrics.classification.BinaryAUROC()
        self.ap_fn = torchmetrics.classification.BinaryAveragePrecision()
        self.spe_fn = torchmetrics.classification.BinarySpecificity()
        self.sen_fn = torchmetrics.classification.BinaryRecall()



    def training_step(self, batch, batch_idx):
        x = batch.x
        y = batch.y
        edge_index = batch.edge_index

        if batch.edge_attr is not None:
            edge_attr = batch.edge_attr

        x = self.feature_extractor(x)
        x = self.gnn(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch.batch)  

        pred = self.classify(x, clinical=batch.clinical)

        loss = self.loss_fn(pred, y)

        self.auc_fn(pred, y) 
        self.ap_fn(pred, y) 
        self.sen_fn(pred, y) 
        self.spe_fn(pred, y) 

        self.log("train_loss_step", loss, on_epoch=True)
        self.log("train_auc_step", self.auc_fn, on_epoch=True)
        self.log("train_ap_step", self.ap_fn, on_epoch=True)
        self.log("train_sen_step", self.sen_fn, on_epoch=True)
        self.log("train_spe_step", self.spe_fn, on_epoch=True)


    def validation_step(self, batch, batch_idx):
        x = batch.x
        y = batch.y
        edge_index = batch.edge_index

        if batch.edge_attr is not None:
            edge_attr = batch.edge_attr

        x = self.feature_extractor(x)
        x = self.gnn(x, edge_index, batch.batch)  

        pred = self.classify(x)

        val_loss = self.loss_fn(pred, y)

        self.auc_fn(pred, y) 
        self.ap_fn(pred, y) 
        self.sen_fn(pred, y) 
        self.spe_fn(pred, y) 

        self.log("val_loss", val_losa=s)
        self.log("val_auc", self.auc_fn)
        self.log("val_ap", self.ap_fn)
        self.log("val_sen", self.sen_fn)
        self.log("val_spe", self.spe_fn)


    def test_step(self, batch, batch_idx):
        x = batch.x
        y = batch.y
        edge_index = batch.edge_index

        if batch.edge_attr is not None:
            edge_attr = batch.edge_attr

        x = self.feature_extractor(x)
        x = self.gnn(x, edge_index, batch.batch)  

        pred = self.classify(x)

        test_loss = self.loss_fn(pred, y)

        self.auc_fn(pred, y) 
        self.ap_fn(pred, y) 
        self.sen_fn(pred, y) 
        self.spe_fn(pred, y) 

        self.log("test_auc", self.auc_fn)
        self.log("test_ap", self.ap_fn)
        self.log("test_sen", self.sen_fn)
        self.log("test_spe", self.spe_fn)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            "interval": "epoch",
            "frequency": self.config['lr_patience'],
            "monitor": "val_loss",
            "strict": True,
            "name": None,
        }

        return {'optimizer': optimizer,
                'lr_scheduler': lr_scheduler_config,}


class CNN(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.learning_rate = self.config['learning_rate']
        self.extractor = getattr(en, self.config['extractor_name'])(n_classes=self.config['n_extracted_channels'], in_channels=self.config['n_in_channels'], dropout=self.config['dropout'])
 
        self.classify = Classify(n_classes=self.config['n_classes'])

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.auc_fn = torchmetrics.classification.BinaryAUROC()
        self.ap_fn = torchmetrics.classification.BinaryAveragePrecision()
        self.spe_fn = torchmetrics.classification.BinarySpecificity()
        self.sen_fn = torchmetrics.classification.BinaryRecall()



    def training_step(self, batch, batch_idx):
        x = batch.x
        y = batch.y

        x = self.feature_extractor(x)

        pred = self.classify(x, clinical=batch.clinical)

        loss = self.loss_fn(pred, y)

        self.auc_fn(pred, y) 
        self.ap_fn(pred, y) 
        self.sen_fn(pred, y) 
        self.spe_fn(pred, y) 

        self.log("train_loss_step", loss, on_epoch=True)
        self.log("train_auc_step", self.auc_fn, on_epoch=True)
        self.log("train_ap_step", self.ap_fn, on_epoch=True)
        self.log("train_sen_step", self.sen_fn, on_epoch=True)
        self.log("train_spe_step", self.spe_fn, on_epoch=True)


    def validation_step(self, batch, batch_idx):
        x = batch.x
        y = batch.y

        x = self.feature_extractor(x)

        pred = self.classify(x)

        val_loss = self.loss_fn(pred, y)

        self.auc_fn(pred, y) 
        self.ap_fn(pred, y) 
        self.sen_fn(pred, y) 
        self.spe_fn(pred, y) 

        self.log("val_loss", val_losa=s)
        self.log("val_auc", self.auc_fn)
        self.log("val_ap", self.ap_fn)
        self.log("val_sen", self.sen_fn)
        self.log("val_spe", self.spe_fn)


    def test_step(self, batch, batch_idx):
        x = batch.x
        y = batch.y

        x = self.feature_extractor(x)

        pred = self.classify(x)

        test_loss = self.loss_fn(pred, y)

        self.auc_fn(pred, y) 
        self.ap_fn(pred, y) 
        self.sen_fn(pred, y) 
        self.spe_fn(pred, y) 

        self.log("test_auc", self.auc_fn)
        self.log("test_ap", self.ap_fn)
        self.log("test_sen", self.sen_fn)
        self.log("test_spe", self.spe_fn)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            "interval": "epoch",
            "frequency": self.config['lr_patience'],
            "monitor": "val_loss",
            "strict": True,
            "name": None,
        }

        return {'optimizer': optimizer,
                'lr_scheduler': lr_scheduler_config,}
