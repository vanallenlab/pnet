from pnet import pnet_loader, Pnet
from util import util, sankey_diag
import torch
import pandas as pd
import numpy as np
import random
import pickle
import os


def main():
    print("Testing Dataloader \n")

    # rna_ext_val = pd.read_csv('/mnt/disks/pancan/data/mel_dfci_2019/data_RNA_Seq_expression_tpm_all_sample_Zscores.txt',
    #                           delimiter='\t').set_index('Hugo_Symbol').T.drop('Entrez_Gene_Id').dropna(axis=1)
    # cna_ext_val = pd.read_csv('/mnt/disks/pancan/data/mel_dfci_2019/data_CNA.txt',
    #                           delimiter='\t').set_index('Hugo_Symbol').T.dropna(axis=1)
    # ext_val = pd.read_csv('/mnt/disks/pancan/data/mel_dfci_2019/data_clinical_sample.txt',
    #                              delimiter='\t').set_index('Sample Identifier').iloc[4:]
    # important_genes = list(pd.read_csv('/mnt/disks/pancan/m1000/cancer_genes.txt')['genes'].values)
    # joint_genes = list(set(important_genes).intersection(list(rna_ext_val.columns), list(cna_ext_val.columns)))
    # gene_list = random.sample(joint_genes, 500)
    # random_genes_a = list(rna_ext_val.sample(5, axis=1).columns)
    # random_genes_b = list(cna_ext_val.sample(5, axis=1).columns)
    # joint_samples = list(rna_ext_val.sample(20).join(cna_ext_val, rsuffix='_cna', how='inner').index)
    # random_samples_a = list(rna_ext_val.sample(5, axis=0).index)
    # random_samples_b = list(cna_ext_val.sample(5, axis=0).index)
    # random_samples_c = list(cna_ext_val.sample(5, axis=0).index)
    # random_samples_d = list(cna_ext_val.sample(5, axis=0).index)
    # test_rna = rna_ext_val.loc[joint_samples+random_samples_a][joint_genes+random_genes_a].copy().drop_duplicates()
    # test_cna = cna_ext_val.loc[joint_samples+random_samples_b][joint_genes+random_genes_b].copy().drop_duplicates()
    # test_add = ext_val.loc[joint_samples+random_samples_c][['Purity', 'Ploidy']].copy().drop_duplicates()
    # test_y = ext_val.loc[joint_samples+random_samples_d][['Heterogeneity']].copy().drop_duplicates()
    # test_rna.reset_index(inplace=True)
    # test_cna.reset_index(inplace=True)
    # test_add.reset_index(inplace=True)
    # test_y.reset_index(inplace=True)
    # test_rna.rename(columns={'index': 'sample_id'}, inplace=True)
    # test_cna.rename(columns={'index': 'sample_id'}, inplace=True)
    # test_add.rename(columns={'Sample Identifier': 'sample_id'}, inplace=True)
    # test_y.rename(columns={'Sample Identifier': 'sample_id'}, inplace=True)
    # test_rna.to_csv('../data/test_data/rna.csv', index=False)
    # test_cna.to_csv('../data/test_data/cna.csv', index=False)
    # test_add.to_csv('../data/test_data/add.csv', index=False)
    # test_y.to_csv('../data/test_data/y.csv', index=False)
    # with open('../data/test_data/gene_sublist.txt', 'wb') as fp:
    #     pickle.dump(gene_list, fp)

    test_rna = pd.read_csv('data/test_data/rna.csv').set_index('sample_id')
    test_cna = pd.read_csv('data/test_data/cna.csv').set_index('sample_id')
    test_add = pd.read_csv('data/test_data/add.csv').set_index('sample_id')
    test_y = pd.read_csv('data/test_data/y.csv').set_index('sample_id')

    with open('data/test_data/gene_sublist.txt', 'rb') as fp:
        gene_list = pickle.load(fp)

    genetic_data = {'rna': test_rna, 'cna': test_cna}

    train_dataset, test_dataset = pnet_loader.generate_train_test(genetic_data,
                                                                  test_y, 
                                                                  additional_data=test_add,
                                                                  test_split=0.2,
                                                                  gene_set=gene_list,
                                                                  collinear_features=2)

    assert set(gene_list) == set(train_dataset.genes), 'Training dataset expected to have the same gene set as in file'
    assert train_dataset.genes == [x.split('_')[0] for x in list(train_dataset.input_df.columns)[:500]], 'Training data genes should be ordered \
                                                                                as stored in the genes variable'
    assert train_dataset.input_df.shape == torch.Size([16, 1000]), 'Input DataFrame expected to be a of size\
                                                            [16, 1000], got: {}'.format(train_dataset.input_df.shape)
    assert train_dataset.x.shape == torch.Size([16, 1000]), 'Small train dataset expected to be a tensor of size\
                                                            [16, 1000], got: {}'.format(train_dataset.x.shape)
    assert train_dataset.y.shape == torch.Size([16, 1]), 'Small train dataset expected to be a tensor of size\
                                                            [16, 1], got: {}'.format(train_dataset.y.shape)
    
    print('All done, all tests passed!')

if __name__ == "__main__":
    main()


