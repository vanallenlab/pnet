import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC
import numpy as np
import torch.optim as optim
import pytorch_lightning as pl
import ReactomeNetwork
import pnet_loader
import Pnet
from CustomizedLinear import masked_activation
import util


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


def fit(model, dataloader, optimizer):
    pred_loss = nn.BCELoss(reduction='sum')
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for batch in dataloader:
        gene_data, additional_data, y = batch
        optimizer.zero_grad()
        y_hat = model(gene_data, additional_data)
        loss = pred_loss(torch.squeeze(y_hat), torch.squeeze(y))
        acc = np.sum(y_hat.round().detach().numpy().squeeze() == y.detach().numpy().squeeze())
        running_loss += loss.item()
        running_acc += acc
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    train_acc = running_acc/len(dataloader.dataset)
    return train_loss, train_acc


def validate(model, dataloader):
    pred_loss = nn.BCELoss(reduction='sum')
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    for batch in dataloader:
        gene_data, additional_data, y = batch
        y_hat = model(gene_data, additional_data)
        loss = pred_loss(torch.squeeze(y_hat), torch.squeeze(y))
        acc = np.sum(y_hat.round().detach().numpy().squeeze() == y.detach().numpy().squeeze())
        running_loss += loss.item()
        running_acc += acc
        loss.backward()
    loss = running_loss / len(dataloader.dataset)
    acc = running_acc / len(dataloader.dataset)
    return loss, acc


def train(model, train_loader, test_loader, lr=0.5e-4, weight_decay=1e-5, epochs=300, verbose=False):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopper = util.EarlyStopper(patience=5, min_delta=0.01)
    train_scores = {'loss':[], 'acc':[]}
    test_scores = {'loss':[], 'acc':[]}
    for epoch in range(epochs):
        train_epoch_scores = fit(model, train_loader, optimizer)
        test_epoch_scores = validate(model, test_loader)
        for i, k in enumerate(train_scores):
            train_scores[k].append(train_epoch_scores[i])
            test_scores[k].append(test_epoch_scores[i])
        if verbose:
            print(f"Epoch {epoch + 1} of {epochs}")
            print("Train scores: {}".format(train_epoch_scores))
            print("Test scores: {}".format(test_epoch_scores))
        if early_stopper.early_stop(test_epoch_scores[0]):
            print('Hit early stopping criteria')
            break
    return model, train_scores, test_scores


def run(genetic_data, target, gene_set=None, additional_data=None, test_split=0.2, seed=None, dropout=0.2,
        lr=0.5e-4, weight_decay=1e-4, batch_size=64, epochs=300, verbose=False):
    train_dataset, test_dataset = pnet_loader.generate_train_test(genetic_data, target, gene_set, additional_data,
                                                                  test_split, seed)
    reactome_network = ReactomeNetwork.ReactomeNetwork(train_dataset.get_genes())
    model = PNET_NN(hparams=
                    {'reactome_network':reactome_network, 'nbr_gene_inputs':len(genetic_data), 'dropout':dropout,
                      'additional_dims':train_dataset.additional_data.shape[1], 'lr':lr, 'weight_decay':weight_decay}
                    )
    train_loader, test_loader = pnet_loader.to_dataloader(train_dataset, test_dataset, batch_size)
    model, train_scores, test_scores = train(model, train_loader, test_loader, lr, weight_decay, epochs, verbose)
    return model, train_scores, test_scores

