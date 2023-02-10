import pandas as pd
import numpy as np


def select_highly_variable_genes(df, bins=10, genes_per_bin=100):
    if len(df.columns) < bins*genes_per_bin:
        raise ValueError('Want to select more genes than present in Input')
    bin_assignment = pd.DataFrame(pd.qcut(df.sum(axis=0), 10, labels=False, duplicates='drop'), columns=['bin'])
    gene_std = pd.DataFrame(df.std(), columns=['std'])
    return bin_assignment.join(gene_std).groupby('bin')['std'].nlargest(genes_per_bin).reset_index()


def select_non_constant_genes(df, cutoff=0.05):
    """
    Many expression datasets have genes that have mostly constant expression throughout the dataset, we can select for
    genes that have minimum percentage of unique values.
    :param df: pd.DataFrame; dataframe to select columns from
    :param cutoff: float; percentage of unique values we require
    :return: pd.DataFrame; cleaned dataframe with only non-constant columns
    """
    return df.loc[:, df.nunique()/df.count() > 0.05].copy()


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_validation_loss*self.min_delta):
            self.counter += 1
            print('exceeded delta')
            if self.counter >= self.patience:
                return True
        return False
