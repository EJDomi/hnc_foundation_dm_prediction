import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

import torchvision
import torchmetrics

import pytorch_lightning as L

from lifelines.utils import concordance_index
import hnc_project.pytorch.extractor_networks as en
import hnc_project.pytorch.gnn_networks as graphs
import hnc_project.pytorch.user_metrics as um

class Classify(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.classify = nn.Linear(in_channels, n_classes)
        #self.classify = nn.LazyLinear(n_classes)


    def forward(self, x):


        x = self.classify(x)
        # the following is get the shape right so pytorch doesn't yell at you, 
        # in the off chance that the batch only has 1 entry
        if len(x) == 1:
            x = x.squeeze().unsqueeze(0)
        else:
            x = x.squeeze()
        return x


class CNN_GNN(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer("clinical_mean", torch.tensor(np.mean(self.config['clinical_means']['RADCURE'], axis=0)))
        self.register_buffer("clinical_std", torch.tensor(np.mean(self.config['clinical_stds']['RADCURE'], axis=0)))

        self.learning_rate = self.config['learning_rate']

        gnn_in_channels = self.config['extractor_channels']

        if self.config['use_radiomics'] and (config['extractor_name'] == 'EmptyNetwork' or config['use_images']):
            gnn_in_channels += self.config['n_radiomics']

        if self.config['use_embeddings'] and (config['extractor_name'] == 'EmptyNetwork' or config['use_images']):
            gnn_in_channels += self.config['n_embeddings']
        #if self.config['use_clinical']:
        #    gnn_in_channels += self.config['n_clinical']

        self.extractor = getattr(en, self.config['extractor_name'])(n_classes=self.config['extractor_channels'], in_channels=self.config['n_in_channels'], dropout=self.config['dropout'])
        self.gnn = getattr(graphs, self.config['model_name'])(gnn_in_channels, hidden_channels=self.config['n_hidden_channels'], n_classes=self.config['n_hidden_channels'], edge_dim=self.config['edge_dim'], dropout=self.config['dropout'])
       
        in_channels = self.config['n_hidden_channels']
        if self.config['use_clinical']:
            in_channels += self.config['n_clinical']
      
        if self.config['multi_label'] and self.config['n_classes'] != 4:
            raise Exception('number of classes too small for multi-label')

        self.classify = Classify(in_channels=in_channels, n_classes=self.config['n_classes'])

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.config['class_weight']]))
        #self.val_loss_fn = nn.BCEWithLogitsLoss()
        #self.test_loss_fn = nn.BCEWithLogitsLoss()

        self.m_fn = um.MMetric(0.6, 0.4)
        if self.config['multi_label']:
            self.auc_fn = torchmetrics.classification.MultilabelAUROC(num_labels=self.config['n_classes'], average='micro')
            self.ap_fn = torchmetrics.classification.MultilabelAveragePrecision(num_labels=self.config['n_classes'], average='micro')
            self.spe_fn = torchmetrics.classification.MultilabelSpecificity(num_labels=self.config['n_classes'], average='micro')
            self.sen_fn = torchmetrics.classification.MultilabelRecall(num_labels=self.config['n_classes'], average='micro')

            self.val_auc_fn = torchmetrics.classification.MultilabelAUROC(num_labels=self.config['n_classes'], average='micro')
            self.val_ap_fn = torchmetrics.classification.MultilabelAveragePrecision(num_labels=self.config['n_classes'], average='micro')
            self.val_spe_fn = torchmetrics.classification.MultilabelSpecificity(num_labels=self.config['n_classes'], average='micro')
            self.val_sen_fn = torchmetrics.classification.MultilabelRecall(num_labels=self.config['n_classes'], average='micro')

            self.test_auc_fn = torchmetrics.classification.MultilabelAUROC(num_labels=self.config['n_classes'], average='micro')
            self.test_ap_fn = torchmetrics.classification.MultilabelAveragePrecision(num_labels=self.config['n_classes'], average='micro')
            self.test_spe_fn = torchmetrics.classification.MultilabelSpecificity(num_labels=self.config['n_classes'], average='micro')
            self.test_sen_fn = torchmetrics.classification.MultilabelRecall(num_labels=self.config['n_classes'], average='micro')
        else:
            self.auc_fn = torchmetrics.classification.BinaryAUROC()
            self.ap_fn = torchmetrics.classification.BinaryAveragePrecision()
            self.spe_fn = torchmetrics.classification.BinarySpecificity()
            self.sen_fn = torchmetrics.classification.BinaryRecall()

            self.val_auc_fn = torchmetrics.classification.BinaryAUROC()
            self.val_ap_fn = torchmetrics.classification.BinaryAveragePrecision()
            self.val_spe_fn = torchmetrics.classification.BinarySpecificity()
            self.val_sen_fn = torchmetrics.classification.BinaryRecall()

            self.test_auc_fn = torchmetrics.classification.BinaryAUROC()
            self.test_ap_fn = torchmetrics.classification.BinaryAveragePrecision()
            self.test_spe_fn = torchmetrics.classification.BinarySpecificity()
            self.test_sen_fn = torchmetrics.classification.BinaryRecall()
        self.test_preds = []
        self.test_targets = []
        self.save_hyperparameters()

    def init_params(self, m):
        """
           Following is the doc string from stolen function:
                Initialize the parameters of a module.
                Parameters
                ----------
                m
                    The module to initialize.
                Notes
                -----
                Convolutional layer weights are initialized from a normal distribution
                as described in [1]_ in `fan_in` mode. The final layer bias is
                initialized so that the expected predicted probability accounts for
                the class imbalance at initialization.
                References
                ----------
                .. [1] K. He et al. ‘Delving Deep into Rectifiers: Surpassing
                   Human-Level Performance on ImageNet Classification’,
                   arXiv:1502.01852 [cs], Feb. 2015.
        """
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=.1)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0.)


    def _shared_eval_step(self, batch, batch_idx):
        with torch.no_grad():
            if self.config['use_embeddings'] and self.config['use_radiomics'] and not self.config['use_images']:
                x = batch.embeddings
                x = torch.cat((x, batch.radiomics), 1)
            elif self.config['use_embeddings'] and not self.config['use_images']:
                x = batch.embeddings
            elif self.config['use_radiomics'] and not self.config['use_images']:
                x = batch.radiomics
            else:
                x = batch.x
            edge_index = batch.edge_index


            if batch.edge_attr is not None and self.config['edge_dim'] is not None:
                edge_attr = batch.edge_attr.float()
            else:
                edge_attr = None

            if 'vit' in self.config['extractor_name']:
                x = self.extractor(x)[0]
                #x = self.avg_pool(x)
            else:
                x = self.extractor(x)
 
            if x.dim() == 1:
                x = x.squeeze().unsqueeze(0)

            if self.config['use_embeddings'] and self.config['use_images']:
                x = torch.cat((x, batch.embeddings), 1)

            if self.config['use_radiomics'] and self.config['use_images']:
                x = torch.cat((x,batch.radiomics), 1)

            #if self.config['use_clinical']:
            #    clinical = batch.clinical
            #    clinical[:, 0:4] = (batch.clinical[:, 0:4] - self.clinical_mean) / self.clinical_std
            #else:
            #    clinical = None
            #if clinical is not None:
            #    #clinical = torch.unique(clinical, dim=0)
            #    x = torch.cat((x, clinical), 1)
            x = self.gnn(x=x, edge_index=edge_index, batch=batch.batch, edge_attr=edge_attr)  

            if self.config['graph_pooling']:
                x = global_mean_pool(x, batch.batch)
            else:
                x = torch.stack(tuple([x[batch.batch==idx][-1] for idx in batch.batch.unique()]))

            if self.config['use_clinical']:
                clinical = batch.clinical
                clinical[:, 0:4] = (batch.clinical[:, 0:4] - self.clinical_mean) / self.clinical_std
            else:
                clinical = None

            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            if clinical is not None:
                #clinical = torch.unique(clinical, dim=0)
                x = torch.cat((x, clinical), 1)

            x = self.classify(x)
        return x
        

    def training_step(self, batch, batch_idx):

        if self.config['use_embeddings'] and self.config['use_radiomics'] and not self.config['use_images']:
            x = batch.embeddings
            x = torch.cat((x, batch.radiomics), 1)
        elif self.config['use_embeddings'] and not self.config['use_images']:
            x = batch.embeddings
        elif self.config['use_radiomics'] and not self.config['use_images']:
            x = batch.radiomics
        else:
            x = batch.x

        edge_index = batch.edge_index
        if batch.edge_attr is not None and self.config['edge_dim'] is not None:
            edge_attr = batch.edge_attr.float()
        else:
            edge_attr = None


        if 'vit' in self.config['extractor_name']:
            x = self.extractor(x)[0]
            #x = self.avg_pool(x)
        else:
            x = self.extractor(x)
        if x.dim() == 1:
            x = x.squeeze().unsqueeze(0)

        if self.config['use_embeddings'] and self.config['use_images']:
            x = torch.cat((x, batch.embeddings), 1)

        if self.config['use_radiomics'] and self.config['use_images']:
            x = torch.cat((x,batch.radiomics), 1)

        #if self.config['use_clinical']:
        #    clinical = batch.clinical
        #    clinical[:, 0:4] = (batch.clinical[:, 0:4] - self.clinical_mean) / self.clinical_std
        #else:
        #    clinical = None

        #if clinical is not None:
        #    #clinical = torch.unique(clinical, dim=0)
        #    x = torch.cat((x, clinical), 1)

        x = self.gnn(x=x, edge_index=edge_index, batch=batch.batch, edge_attr=edge_attr)  

        if self.config['graph_pooling']:
            x = global_mean_pool(x, batch.batch)
        else:
            x = torch.stack(tuple([x[batch.batch==idx][-1] for idx in batch.batch.unique()]))
        
        
            #pred = self(batch, batch_idx) 
        if self.config['use_clinical']:
            clinical = batch.clinical
            clinical[:, 0:4] = (batch.clinical[:, 0:4] - self.clinical_mean) / self.clinical_std
        else:
            clinical = None

        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        if clinical is not None:
            #clinical = torch.unique(clinical, dim=0)
            x = torch.cat((x, clinical), 1)

        pred = self.classify(x)

        if self.config['multi_label']:
            dm_loss = self.loss_fn(pred[:,0], batch.y.to(torch.float))
            lm_loss = self.loss_fn(pred[:,1], batch.lm.to(torch.float))
            rm_loss = self.loss_fn(pred[:,2], batch.rm.to(torch.float))
            death_loss = self.loss_fn(pred[:,3], batch.death.to(torch.float))
            loss = dm_loss + lm_loss + rm_loss + death_loss

            y = torch.cat((batch.y.unsqueeze(0),
                          batch.lm.unsqueeze(0),
                          batch.rm.unsqueeze(0),
                          batch.death.unsqueeze(0))).T

        loss = self.loss_fn(pred, batch.y.to(torch.float))
        y = batch.y


        self.auc_fn(pred, y.to(torch.int64)) 
        self.ap_fn(pred, y.to(torch.int64)) 
        self.sen_fn(pred, y) 
        self.spe_fn(pred, y) 
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=len(batch.batch), prog_bar=True)
        self.log("train_auc", self.auc_fn, on_step=False, on_epoch=True, batch_size=len(batch.batch), prog_bar=True)
        self.log("train_ap", self.ap_fn, on_step=False, on_epoch=True, batch_size=len(batch.batch))
        self.log("train_m", self.m_fn(self.sen_fn.compute(), self.spe_fn.compute()).mean(), on_step=False, on_epoch=True, batch_size=len(batch.batch))
        self.log("train_sen", self.sen_fn, on_step=False, on_epoch=True, batch_size=len(batch.batch))
        self.log("train_spe", self.spe_fn, on_step=False, on_epoch=True, batch_size=len(batch.batch))

        return {'loss': loss, 'dm_loss': dm_loss, 'lm_loss': lm_loss, 'rm_loss': rm_loss, 'death_loss': death_loss} if self.config['multi_label'] else loss

    

    def validation_step(self, batch, batch_idx):
        pred = self._shared_eval_step(batch, batch_idx)

        if self.config['multi_label']:
            dm_val_loss = self.loss_fn(pred[:,0], batch.y.to(torch.float))
            lm_val_loss = self.loss_fn(pred[:,1], batch.lm.to(torch.float))
            rm_val_loss = self.loss_fn(pred[:,2], batch.rm.to(torch.float))
            death_val_loss = self.loss_fn(pred[:,3], batch.death.to(torch.float))
            val_loss = dm_val_loss + lm_val_loss + rm_val_loss + death_val_loss

            y = torch.cat((batch.y.unsqueeze(0),
                          batch.lm.unsqueeze(0),
                          batch.rm.unsqueeze(0),
                          batch.death.unsqueeze(0))).T
        else:
            val_loss = self.loss_fn(pred, batch.y.to(torch.float))
            y = batch.y


        self.val_auc_fn(pred, y.to(torch.int64)) 
        self.val_ap_fn(pred, y.to(torch.int64)) 
        self.val_sen_fn(pred, y) 
        self.val_spe_fn(pred, y) 

        self.log_dict({"val_loss": torch.tensor([val_loss]),
        "val_auc": self.val_auc_fn,
        "val_ap": self.val_ap_fn,
        "val_m": self.m_fn(self.val_sen_fn.compute(), self.val_spe_fn.compute()).mean(),
        "val_sen": self.val_sen_fn,
        "val_spe": self.val_spe_fn,
        }, batch_size=len(batch.batch), prog_bar=True)

        return {"val_loss": val_loss, 'dm_val_loss': dm_val_loss, 'lm_val_loss': lm_val_loss, 'rm_val_loss': rm_val_loss, 'death_val_loss': death_val_loss} if self.config['multi_label'] else {"val_loss": val_loss}


    def test_step(self, batch, batch_idx):
        pred = self._shared_eval_step(batch, batch_idx)
        self.test_preds.append(pred)
        self.test_targets.append(batch.y)

        if self.config['multi_label']:
            dm_test_loss = self.loss_fn(pred[:,0], batch.y.to(torch.float))
            lm_test_loss = self.loss_fn(pred[:,1], batch.lm.to(torch.float))
            rm_test_loss = self.loss_fn(pred[:,2], batch.rm.to(torch.float))
            death_test_loss = self.loss_fn(pred[:,3], batch.death.to(torch.float))
            test_loss = dm_test_loss + lm_test_loss + rm_test_loss + death_test_loss

            y = torch.cat((batch.y.unsqueeze(0),
                          batch.lm.unsqueeze(0),
                          batch.rm.unsqueeze(0),
                          batch.death.unsqueeze(0))).T
        else:
            test_loss = self.loss_fn(pred, batch.y.to(torch.float))
            y = batch.y

        self.test_auc_fn(pred, y.to(torch.int64)) 
        self.test_ap_fn(pred, y.to(torch.int64)) 
        self.test_sen_fn(pred, y) 
        self.test_spe_fn(pred, y) 
        self.log("test_auc", self.test_auc_fn)
        self.log("test_ap", self.test_ap_fn)
        self.log("test_m", self.m_fn(self.test_sen_fn.compute(), self.test_spe_fn.compute()).mean())
        self.log("test_sen", self.test_sen_fn)
        self.log("test_spe", self.test_spe_fn)


    def predict_step(self, batch, batch_idx):
        x = self._shared_eval_step(batch, batch_idx)
        turn = nn.Sigmoid()
        pred = turn(x)
        return pred


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        #optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=0.001, factor=0.1, verbose=True),
            "interval": "epoch",
            "frequency": self.config['lr_patience'],
            "monitor": "val_loss",
            "strict": True,
            "name": None,
        }

        return {'optimizer': optimizer,
                'lr_scheduler': lr_scheduler_config,}
        #return {'optimizer': optimizer,}

