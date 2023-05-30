import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.classification import BinaryAUROC
import numpy as np
import os
import pytorch_lightning as pl
import captum
import ReactomeNetwork
import pnet_loader
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
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, genes):
        x_genes = self.gene_layer(genes)
        x_pathway = self.pathway_layer(x)
        return self.dropout(self.activation(self.batchnorm(x_genes + x_pathway)))


class PNET_NN(pl.LightningModule):

#     @staticmethod
#     def add_model_specific_args(parent_parser):
#         parser = ArgumentParser(parents=[parent_parser], add_help=False)
#         parser.add_argument('--reactome_network', type=ReactomeNetwork.ReactomeNetwork)
#         parser.add_argument('--nbr_gene_inputs', type=int, default=1)
#         parser.add_argument('--additional_dims', type=int, default=0)

#         parser.add_argument('--lr', type=float, default=1e-3)
#         parser.add_argument('--weight_decay', type=float, default=1e-5)
#         parser.add_argument('--dropout', type=float, default=0.2)
#         return parser

    def __init__(self, reactome_network, nbr_gene_inputs=1, output_dim=1, additional_dims=0, lr=1e-3, weight_decay=1e-5, dropout=0.2, random_network=False, fcnn=False):
        super().__init__()
        self.reactome_network = reactome_network
        self.nbr_gene_inputs = nbr_gene_inputs
        self.output_dim = output_dim
        self.additional_dims =additional_dims
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        # Fetch connection masks from reactome network:
        gene_masks, pathway_masks, input_mask = self.reactome_network.get_masks(self.nbr_gene_inputs)
        if random_network:
            for gm in gene_masks:
                util.shuffle_connections(gm)
            for pm in pathway_masks:
                util.shuffle_connections(pm)
                
        if fcnn:
            gene_masks = [np.ones_like(gm) for gm in gene_masks]
            pathway_masks = [np.ones_like(gm) for gm in pathway_masks]
                
        # Prepare list of layers and list of predictions per layer:
        self.layers = nn.ModuleList()
        self.preds = nn.ModuleList()
        # Add input layer to aggregate all data modalities
        self.input_layer = nn.Sequential(*masked_activation(input_mask, activation='tanh'))
        # Add first layer separately:
        self.first_gene_layer = nn.Sequential(*masked_activation(gene_masks[0], activation='tanh'))
        self.drop1 = nn.Dropout(self.dropout)
        # Add blocks and prediction heads for each pathway level:
        for i in range(0, len(gene_masks) - 2):
            self.layers.append(PNET_Block(gene_masks[i + 1], pathway_masks[i], self.dropout))
            self.preds.append(
                nn.Sequential(*[nn.Linear(in_features=pathway_masks[i].shape[0] + self.additional_dims,
                                          out_features=self.output_dim),
                                nn.Sigmoid()]))
        # Add final prediction layer:
        self.preds.append(nn.Sequential(*[nn.Linear(in_features=pathway_masks[len(gene_masks) - 2].shape[0] +
                                                                self.additional_dims, out_features=self.output_dim),
                                          nn.Sigmoid()]))
        # Weighting of the different prediction layers:
        self.attn = nn.Linear(in_features=(len(gene_masks) - 1) * self.output_dim, out_features=self.output_dim)

    def forward(self, x, additional_data):
        x = self.input_layer(x)
        genes = torch.clone(x)
        y_hats = []
        x = self.drop1(self.first_gene_layer(x))
        x_cat = torch.concat([x, additional_data], dim=1)
        y_hats.append(self.preds[0](x_cat))
        for layer, pred in zip(self.layers, self.preds[1:]):
            x = layer(x, genes)
            x_cat = torch.concat([x, additional_data], dim=1)
            y_hats.append(pred(x_cat))
        y = self.attn(torch.concat(y_hats, dim=1))
        return y

    def step(self, who, batch, batch_nb):
        x, additional, y = batch
        pred_y = self(x, additional)
        loss = F.cross_entropy(pred_y, y, reduction='mean')

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
        return torch.optim.Adam(self.parameters(), lr=self.lr)


    def deepLIFT(self, test_dataset, target_class=0):
        dl = captum.attr.DeepLift(self)
        gene_importances, additional_importances = dl.attribute((test_dataset.x, test_dataset.additional)
                                                                , target=target_class)
        gene_importances = pd.DataFrame(gene_importances.detach().numpy(),
                                        index=test_dataset.input_df.index,
                                        columns=test_dataset.input_df.columns)
        additional_importances = pd.DataFrame(additional_importances.detach().numpy(),
                                              index=test_dataset.additional_data.index,
                                              columns=test_dataset.additional_data.columns)
        self.gene_importances, self.additional_importances = gene_importances, additional_importances
        return self.gene_importances, self.additional_importances
    
    def integrated_gradients(self, test_dataset, target_class=0):
        ig = captum.attr.IntegratedGradients(self)
        ig_attr, delta = ig.attribute((test_dataset.x, test_dataset.additional), return_convergence_delta=True
                                      , target=target_class)
        gene_importances, additional_importances = ig_attr
        gene_importances = pd.DataFrame(gene_importances.detach().numpy(),
                                        index=test_dataset.input_df.index,
                                        columns=test_dataset.input_df.columns)
        additional_importances = pd.DataFrame(additional_importances.detach().numpy(),
                                              index=test_dataset.additional_data.index,
                                              columns=test_dataset.additional_data.columns)
        self.gene_importances, self.additional_importances = gene_importances, additional_importances
        return self.gene_importances, self.additional_importances

    def layerwise_importance(self, test_dataset, target_class=0):
        layer_importance_scores = []
        for i, level in enumerate(self.layers):
            cond = captum.attr.LayerConductance(self, level.activation)  # ReLU output of masked layer at each level
            cond_vals = cond.attribute((test_dataset.x, test_dataset.additional), target=target_class)
            cols = [self.reactome_network.pathway_encoding.set_index('ID').loc[col]['pathway'] for col in self.reactome_network.pathway_layers[i].columns]
            cond_vals_genomic = pd.DataFrame(cond_vals.detach().numpy(),
                                             columns=cols,
                                             index=test_dataset.input_df.index)
            pathway_imp_by_target = cond_vals_genomic
            layer_importance_scores.append(pathway_imp_by_target)
        return layer_importance_scores
    
    def gene_importance(self, test_dataset, target_class=0):
        cond = captum.attr.LayerConductance(self, self.input_layer)
        cond_vals = cond.attribute((test_dataset.x, test_dataset.additional), target=target_class)
        cols = self.reactome_network.gene_list
        cond_vals_genomic = pd.DataFrame(cond_vals.detach().numpy(),
                                         columns=cols,
                                         index=test_dataset.input_df.index)
        gene_imp_by_target = cond_vals_genomic
        return gene_imp_by_target

    def interpret(self, test_dataset, plot=False):
        gene_feature_importances, additional_feature_importances = self.integrated_gradients(test_dataset)
        gene_importances = self.gene_importance(test_dataset)
        layer_importance_scores = self.layerwise_importance(test_dataset)
        
        gene_order = gene_importances.mean().sort_values(ascending=True).index
        if plot:
            plt.rcParams["figure.figsize"] = (6,8)
            gene_importances[list(gene_order[-20:])].plot(kind='box', vert=False)
            plt.savefig(plot+'/imp_genes.pdf')
            
        return gene_feature_importances, additional_feature_importances, gene_importances, layer_importance_scores


def fit(model, dataloader, optimizer):
    if model.output_dim == 1:
        pred_loss = nn.BCEWithLogitsLoss(reduction='sum')
    else:
        pred_loss = nn.CrossEntropyLoss(reduction='sum')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        gene_data, additional_data, y = batch
        gene_data, additional_data, y = gene_data.to(device), additional_data.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(gene_data, additional_data)
        loss = pred_loss(y_hat, y)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss


def validate(model, dataloader):
    if model.output_dim == 1:
        pred_loss = nn.BCEWithLogitsLoss(reduction='sum')
    else:
        pred_loss = nn.CrossEntropyLoss(reduction='sum')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model.eval()
    running_loss = 0.0
    for batch in dataloader:
        gene_data, additional_data, y = batch
        gene_data, additional_data, y = gene_data.to(device), additional_data.to(device), y.to(device)
        y_hat = model(gene_data, additional_data)
        loss = pred_loss(torch.squeeze(y_hat), torch.squeeze(y))
        running_loss += loss.item()
        loss.backward()
    loss = running_loss / len(dataloader.dataset)
    return loss


def train(model, train_loader, test_loader, lr=0.5e-3, weight_decay=1e-4, epochs=300, verbose=False,
          early_stopping=True):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('We are sending to cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopper = util.EarlyStopper(patience=5, min_delta=0.01, verbose=verbose)
    train_scores = []
    test_scores = []
    for epoch in range(epochs):
        train_epoch_loss = fit(model, train_loader, optimizer)
        test_epoch_loss = validate(model, test_loader)
        train_scores.append(train_epoch_loss)
        test_scores.append(test_epoch_loss)
        if verbose:
            print(f"Epoch {epoch + 1} of {epochs}")
            print("Train Loss: {}".format(train_epoch_loss))
            print("Test Loss: {}".format(test_epoch_loss))
        if early_stopper.early_stop(test_epoch_loss) and early_stopping:
            print('Hit early stopping criteria')
            break
    return model, train_scores, test_scores


def evaluate_interpret_save(model, test_dataset, path):
    if not os.path.exists(path):
        os.makedirs(path)
    x_test = test_dataset.x
    additional_test = test_dataset.additional
    y_test = test_dataset.y
    model.to('cpu')
    pred = model(x_test, additional_test)
    auc = util.get_auc(pred, y_test, save=path+'/auc_curve.pdf')
    
    torch.save(pred, path+'/predictions.pt')
    torch.save(auc, path+'/AUC.pt')
    gene_feature_importances, additional_feature_importances, gene_importances, layer_importance_scores = model.interpret(test_dataset)
    gene_feature_importances.to_csv(path+'/gene_feature_importances.csv')
    additional_feature_importances.to_csv(path+'/additional_feature_importances.csv')
    gene_importances.to_csv(path+'/gene_importances.csv')
    for i, layer in enumerate(layer_importance_scores):
        layer.to_csv(path+'/layer_{}_importances.csv'.format(i))



def run(genetic_data, target, gene_set=None, additional_data=None, test_split=0.2, seed=None, dropout=0.3,
        lr=1e-3, weight_decay=1, batch_size=64, epochs=300, verbose=False, early_stopping=True, train_inds=None,
        test_inds=None, random_network=False, fcnn=False):
    train_dataset, test_dataset = pnet_loader.generate_train_test(genetic_data, target, gene_set, additional_data,
                                                                  test_split, seed, train_inds, test_inds)
    reactome_network = ReactomeNetwork.ReactomeNetwork(train_dataset.get_genes())
    model = PNET_NN(reactome_network=reactome_network, nbr_gene_inputs=len(genetic_data), dropout=dropout,
                    additional_dims=train_dataset.additional_data.shape[1], lr=lr, weight_decay=weight_decay,
                    output_dim=target.shape[1], random_network=random_network, fcnn=fcnn
                    )
    train_loader, test_loader = pnet_loader.to_dataloader(train_dataset, test_dataset, batch_size)
    model, train_scores, test_scores = train(model, train_loader, test_loader, lr, weight_decay, epochs, verbose,
                                             early_stopping)
    return model, train_scores, test_scores, train_dataset, test_dataset


def interpret(model, x, additional,  plots=False, savedir=''):
    '''
    Function to use DeepLift from Captum on PNET model structure. Generates overall feature importance and layerwise
    results.
    :param model: NN model to predict feature importance on. Assuming PNET structure
    :param data: PnetDataset; data object with samples to use gradients on.
    :return:
    '''
    if plots:
        if savedir:
            if not os.path.exists(savedir):
                os.makedirs(savedir)
        else:
            savedir = os.getcwd()
    feature_importance = dict()
    # Overall feature importance
    ig = IntegratedGradients(model)
    ig_attr, delta = ig.attribute((x, additional), return_convergence_delta=True)
    ig_attr_genomic, ig_attr_additional = ig_attr
    feature_importance['overall_genomic'] = ig_attr_genomic.detach().numpy()
    feature_importance['overall_clinical'] = ig_attr_additional.detach().numpy()
    if plots:
        visualize_importances(test_df.columns[:clinical_index],
                              np.mean(feature_importance['overall_clinical'], axis=0),
                              title="Average Feature Importances",
                              axis_title="Clinical Features")
        plt.savefig('/'.join([ savedir, 'feature_importance_overall_clinical.pdf']))

        visualize_importances(test_df.columns[clinical_index:],
                              np.mean(feature_importance['overall_genomic'], axis=0),
                              title="Average Feature Importances",
                              axis_title="Genomic Features")
        plt.savefig('/'.join([savedir, 'feature_importance_overall_genomic.pdf']))

    # Neurons feature importance
    layer_importance_scores = []
    for level in model.layers:
        cond = LayerConductance(model, level.activation)       # ReLU output of masked layer at each level
        cond_vals = cond.attribute((genomic_input, clinical_input))
        cond_vals_genomic = cond_vals.detach().numpy()
        layer_importance_scores.append(cond_vals_genomic)
    feature_importance['layerwise_neurons_genomic'] = layer_importance_scores
    if plots:
        for i, layer in enumerate(feature_importance['layerwise_neurons_genomic']):
            pathway_names = model.reactome_network.pathway_encoding.set_index('ID')
            pathway_names = pathway_names.loc[model.reactome_network.pathway_layers[i+1].index]['pathway']
            visualize_importances(pathway_names,
                                  np.mean(layer, axis=0),
                                  title="Neurons Feature Importances",
                                  axis_title="Pathway activation Features")
            plt.savefig('/'.join([savedir, 'pathway_neurons_layer_{}_importance.pdf'.format(i)]))

    return feature_importance


def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True, axis_title="Features"):
    x_pos = (np.arange(len(feature_names)))
    if plot:
        plt.figure(figsize=(12,6))
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names, rotation=90)
        plt.xlabel(axis_title)
        plt.title(title)