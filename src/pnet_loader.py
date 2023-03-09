import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import numpy as np


# DataLoader object for pytorch. Constructing single loader for all data input modalities.

class PnetDataset(Dataset):
    def __init__(self, genetic_data, target, indicies, additional_data=None, gene_set=None):
        """
        DataLoader initialization, builds object for pytorch data loading. Handles concatenantion of different
        genetic modalities, connection to target and batching.
        :param genetic_data: Dict(str: pd.DataFrame); requires a dict containing a pd.DataFrame for each data modality
         and the str identifier. Paired samples should have matching indicies across Dataframes.
        :param target: pd.DataFrame or pd.Series; requires a single pandas Dataframe or Series with target variable
         paired per sample index. Target can be binary or continuous.
        :param indicies: list(str); List of sample names to be used for joint Dataset
        :param additional_data: pd.DataFrame; Dataframe with additional information per sample. Sample IDs should match
         genetic data. Per default None is provided
        :param gene_set: list(str); List of genes to be considered, by default all overlapping genes are considered
        """
        assert isinstance(genetic_data, dict), f"input data expected to be a dict, got {type(genetic_data)}"
        for inp in genetic_data:
            assert isinstance(inp, str), f"input data keys expected to be str, got {type(inp)}"
            assert isinstance(genetic_data[inp], pd.DataFrame), f"input data values expected to be a dict, got" \
                                                                f" {type(genetic_data[inp])}"
        self.genetic_data = genetic_data
        self.target = target
        self.gene_set = gene_set
        self.altered_inputs = []
        self.inds = indicies
        if additional_data:
            self.additional_data = additional_data
        else:
            self.additional_data = pd.DataFrame(index=self.inds)    # create empty dummy dataframe if no additional data
        self.target = self.target.loc[self.inds]
        self.genes = self.get_genes()
        self.input_df = self.unpack_input()

        assert self.input_df.index.equals(self.target.index)

    def __len__(self):
        return self.input_df.shape[0]

    def __getitem__(self, index):
        x = torch.tensor(self.input_df.iloc[index], dtype=torch.float)
        y = torch.tensor(self.target.iloc[index], dtype=torch.float)
        additional = torch.tensor(self.additional_data.iloc[index], dtype=torch.float)
        return x, additional, y

    def get_genes(self):
        """
        Generate list of genes which are present in all data modalities and in the list of genes to be considered
        :return: List(str); List of gene names
        """
        # drop duplicated columns:
        for inp in self.genetic_data:
            self.genetic_data[inp] = self.genetic_data[inp].loc[:, ~self.genetic_data[inp].columns.duplicated()].copy()
        gene_sets = [set(self.genetic_data[inp].columns) for inp in self.genetic_data]
        if self.gene_set:
            gene_sets.append(self.gene_set)
        genes = list(set.intersection(*gene_sets))
        print('Found {} overlapping genes'.format(len(genes)))
        return genes

    def unpack_input(self):
        """
        Unpacks data modalities into one joint pd.DataFrame. Suffixing gene names by their modality name.
        :return: pd.DataFrame; containing n*m columns, where n is the number of modalities and m the number of genes
        considered.
        """
        input_df = pd.DataFrame(index=self.inds)
        for inp in self.genetic_data:
            input_df = input_df.join(self.genetic_data[inp][self.genes], how='inner', rsuffix='_' + inp)
        print('generated input DataFrame of size {}'.format(input_df.shape))
        return input_df.loc[self.inds]


def get_indicies(genetic_data, target, additional_data=None):
    """
    Generates a list of indicies which are present in all data modalities. Drops duplicated indicies.
    :param genetic_data: Dict(str: pd.DataFrame); requires a dict containing a pd.DataFrame for each data modality
         and the str identifier. Paired samples should have matching indicies across Dataframes.
    :param target: pd.DataFrame or pd.Series; requires a single pandas Dataframe or Series with target variable
        paired per sample index. Target can be binary or continuous.
    :param additional_data: pd.DataFrame; Dataframe with additional information per sample. Sample IDs should match
     genetic data.
    :return: List(str); List of sample names found in all data modalities
    """
    ind_sets = [set(genetic_data[inp].index.drop_duplicates(keep=False)) for inp in genetic_data]
    ind_sets.append(target.index.drop_duplicates(keep=False))
    if additional_data:
        ind_sets.append(additional_data.index.drop_dublicates(keep=False))
    inds = list(set.intersection(*ind_sets))
    print('Found {} overlapping indicies'.format(len(inds)))
    return inds


def generate_train_test(genetic_data, target, gene_set=None, additional_data=None, test_split=0.3, seed=None):
    """
    Takes all data modalities to be used and generates a train and test DataSet with a given split.
    :param genetic_data: Dict(str: pd.DataFrame); requires a dict containing a pd.DataFrame for each data modality
         and the str identifier. Paired samples should have matching indicies across Dataframes.
    :param target: pd.DataFrame or pd.Series; requires a single pandas Dataframe or Series with target variable
        paired per sample index. Target can be binary or continuous.
    :param gene_set: List(str); List of genes to be considered, default is None and considers all genes found in every
        data modality.
    :param additional_data: pd.DataFrame; Dataframe with additional information per sample. Sample IDs should match
    :param test_split: float; Fraction of samples to be used for testing.
    :param seed: int; Random seed to be used for train/test splits.
    :return:
    """
    print('Given {} Input modalities'.format(len(genetic_data)))
    inds = get_indicies(genetic_data, target)
    random.seed(seed)
    random.shuffle(inds)
    train_inds = inds[:int((len(inds) + 1) * (1 - test_split))]
    test_inds = inds[int((len(inds) + 1) * (1 - test_split)):]
    print('Initializing Train Dataset')
    train_dataset = PnetDataset(genetic_data, target, train_inds, additional_data=additional_data, gene_set=gene_set)
    print('Initializing Test Dataset')
    test_dataset = PnetDataset(genetic_data, target, test_inds, additional_data=additional_data, gene_set=gene_set)
    # couple lines to add some genes with a signal perfectly correlated with the target
    # for n in range(2):
    #     r = random.randint(0, len(train_dataset.input_df.columns))
    #     altered_input_col = train_dataset.input_df.columns[r]
    #     train_dataset.altered_inputs.append(altered_input_col)
    #     test_dataset.altered_inputs.append(altered_input_col)
    #     print('set {} in input to target variable'.format(altered_input_col))
    #     train_dataset.input_df[train_dataset.input_df.columns[r]] = train_dataset.target
    #     test_dataset.input_df[test_dataset.input_df.columns[r]] = test_dataset.target
    return train_dataset, test_dataset


def to_dataloader(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                              persistent_workers=True, pin_memory=True,)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                            persistent_workers=True, pin_memory=True,)
    return train_loader, val_loader
