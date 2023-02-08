import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import ReactomeNetwork
from CustomizedLinear import masked_activation


class PNET_Block(nn.Module):
    def __init__(self, gene_mask, pathway_mask, dropout):
        """
        Pathway level unit of deep network. Taking in connections from the gene level and the previous pathway level.
        Combines the two inputs by addition, applies a batchnorm, non-linearity and dropout before passing it to the
        higher order pathway level.
        :param gene_mask: np.array; binary adjacency matrix from gene level to pathways in layer
        :param pathway_mask: np.array; binary adjacency matrix from previous layer to pathways in current layer
        :param dropout: float; fraction of connections to randomly drop out, applied on layer output
        """
        super(PNET_Block, self).__init__()
        self.gene_layer = nn.Sequential(*masked_activation(gene_mask))
        self.pathway_layer = nn.Sequential(*masked_activation(pathway_mask))
        self.batchnorm = nn.BatchNorm1d(gene_mask.shape[1])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, genes):
        x_genes = self.gene_layer(genes)
        x_pathway = self.pathway_layer(x)
        return self.dropout(self.activation(self.batchnorm(x_genes + x_pathway)))


class PNET_NN(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--reactome_network', type=ReactomeNetwork.ReactomeNetwork)
        parser.add_argument('--nbr_gene_inputs', type=int, default=1)
        parser.add_argument('--additional_dims', type=int, default=0)

        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--dropout', type=float, default=0.2)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        # Fetch connection masks from reactome network:
        gene_masks, pathway_masks = self.hparams.reactome_network.get_masks(self.hparams.nbr_gene_inputs)
        # Prepare list of layers and list of predictions per layer:
        self.layers = nn.ModuleList()
        self.preds = nn.ModuleList()
        # Add first layer separately:
        self.first_gene_layer = nn.Sequential(*masked_activation(gene_masks[0], activation='relu'))
        self.drop1 = nn.Dropout(self.hparams.dropout)
        # Add blocks and prediction heads for each pathway level:
        for i in range(0, len(gene_masks) - 2):
            self.layers.append(PNET_Block(gene_masks[i + 1], pathway_masks[i], self.hparams.dropout))
            self.preds.append(
                nn.Sequential(*[nn.Linear(in_features=pathway_masks[i].shape[0] + self.hparams.additional_dims,
                                          out_features=1),
                                nn.ReLU()]))
        # Add final prediction layer:
        self.preds.append(nn.Sequential(*[nn.Linear(in_features=pathway_masks[len(gene_masks) - 2].shape[0] +
                                                                self.hparams.additional_dims, out_features=1),
                                          nn.ReLU()]))
        # Weighting of the different prediction layers:
        self.attn = nn.Linear(in_features=len(gene_masks) - 1, out_features=1)

    def forward(self, x, additional_data):
        genes = torch.clone(x).detach()
        y_hats = []
        x = self.drop1(F.relu(self.first_gene_layer(x)))
        x_cat = torch.concat([x, additional_data], dim=1)
        y_hats.append(self.preds[0](x_cat))
        for layer, pred in zip(self.layers, self.preds[1:]):
            x = layer(x, genes)
            x_cat = torch.concat([x, additional_data], dim=1)
            y_hats.append(pred(x_cat))
        y = torch.sigmoid(self.attn(torch.concat(y_hats, dim=1)))
        return y

    def step(self, who, batch, batch_nb):
        x, additional, y = batch
        pred_y = self(x, additional)
        loss = F.binary_cross_entropy(pred_y, y, reduction='mean')

        self.log(who + '_bce_loss', loss)
        return loss

    def training_step(self, batch, batch_nb):
        # REQUIRED
        loss = self.step('train', batch, batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step('val', batch, batch_nb)
        return loss

    def test_step(self, batch, batch_nb):
        loss = self.step('test', batch, batch_nb)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
