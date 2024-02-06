import pnet_loader
import util
import Pnet
import ReactomeNetwork
import torch
import random
import seaborn as sns
import pandas as pd
import os
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import torch.nn.functional as F
import torch.nn as nn

def main():
    prostate_mutations = pd.read_csv('/mnt/disks/pancan/pnet_database/prostate/processed/P1000_final_analysis_set_cross_important_only.csv')
    prostate_mutations.set_index('Tumor_Sample_Barcode', inplace=True)

    prostate_cnv = pd.read_csv('/mnt/disks/pancan/pnet_database/prostate/processed/P1000_data_CNA_paper.csv')
    prostate_cnv.rename(columns={"Unnamed: 0": "Tumor_Sample_Barcode"}, inplace=True)
    prostate_cnv.set_index('Tumor_Sample_Barcode', inplace=True)

    prostate_response = pd.read_csv('/mnt/disks/pancan/pnet_database/prostate/processed/response_paper.csv')
    prostate_response.rename(columns={'id': "Tumor_Sample_Barcode"}, inplace=True)
    prostate_response.set_index('Tumor_Sample_Barcode', inplace=True)

    prostate_genes = pd.read_csv('/mnt/disks/pancan/pnet_database/genes/tcga_prostate_expressed_genes_and_cancer_genes.csv')
    prostate_genes = list(set(prostate_genes['genes']).intersection(set(prostate_mutations.columns)).intersection(set(prostate_cnv.columns)))

    prostate_cnv = prostate_cnv[prostate_genes].copy()
    prostate_mutations = prostate_mutations[prostate_genes].copy()

    prostate_mutations = prostate_mutations[list(set(prostate_mutations.columns).intersection(prostate_genes))].copy()
    prostate_cnv = prostate_cnv[list(set(prostate_cnv.columns).intersection(prostate_genes))].copy()

    # Regenerate input as specified in prostate_paper
    prostate_mutations = (prostate_mutations > 0).astype(int)
    prostate_amp = (prostate_cnv > 1).astype(int)
    prostate_del = (prostate_cnv < -1).astype(int)

    genetic_data = {'mut': prostate_mutations, 'amp': prostate_amp, 'del': prostate_del}

    canc_genes = list(pd.read_csv('../../pnet_database/genes/cancer_genes.txt').values.reshape(-1))


    weight_decay_values = [5e-3, 1e-2]
    input_dropout_values = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]
    pathway_dropout_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    learning_rate_values = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]
    batch_size_values = [32, 64, 128]

    class_weights = util.get_class_weights(torch.tensor(prostate_response.values).view(-1))
    task = util.get_task(prostate_response)
    target = util.format_target(prostate_response, task)

    train_inds = list(pd.read_csv('../data/splits/train_set_{}.csv'.format(2))['indicies'])
    test_inds = list(pd.read_csv('../data/splits/test_set_{}.csv'.format(2))['indicies'])
    train_dataset, test_dataset = pnet_loader.generate_train_test(genetic_data, target=target, train_inds=train_inds,
                                                                  test_inds=test_inds, gene_set=canc_genes, seed=123)


    x_test = test_dataset.x
    additional_test = test_dataset.additional
    y_test = test_dataset.y

    reactome_network = ReactomeNetwork.ReactomeNetwork(train_dataset.get_genes())
    loss_fn=nn.BCEWithLogitsLoss(reduce=None)

    for weight_decay in weight_decay_values:
        for inp_drop in input_dropout_values:
            for dropout in pathway_dropout_values:
                for lr in learning_rate_values:
                    for batch_size in batch_size_values:
                        save_path = '../results/hyperparam_search/wd{}_id{}_do{}_lr{}_bs{}'.format(weight_decay, inp_drop,
                                                                                                    dropout, lr, batch_size)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        model = Pnet.PNET_NN(reactome_network=reactome_network, task=task, nbr_gene_inputs=len(genetic_data),
                                        dropout=dropout, additional_dims=train_dataset.additional_data.shape[1], lr=lr, 
                                        weight_decay=weight_decay, output_dim=target.shape[1], random_network=False,
                                        fcnn=False, loss_fn=loss_fn, loss_weight=class_weights, gene_dropout=dropout,
                                             input_dropout=inp_drop)

                        train_loader, test_loader = pnet_loader.to_dataloader(train_dataset, test_dataset, batch_size)
                        model, train_scores, test_scores = Pnet.train(model, train_loader, test_loader, save_path+'/model', lr,
                                                                 weight_decay, epochs=400, verbose=False, 
                                                                 early_stopping=True)

                        df = pd.DataFrame(index=['train_loss', 'test_loss'], data=[train_scores, test_scores]).transpose()

                        model.to('cpu')
                        pred, preds = model(x_test, additional_test)
                        y_pred_proba = model.predict_proba(x_test, additional_test).detach()
                        fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
                        test_auc = metrics.roc_auc_score(y_test, y_pred_proba)
                        df['auc'] = test_auc

                        df.to_csv(save_path+'/loss.csv', index=False)
                        
                        
if __name__ == "__main__":
    main()