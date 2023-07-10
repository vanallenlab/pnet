import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import torchmetrics
import torch
import torch.nn as nn
import torch.nn.functional as F

MUTATIONS_DICT = {"3'Flank": 'Silent',
                  "5'Flank": 'Silent',
                  "5'UTR": 'Silent',
                  "3'UTR": 'Silent',
                  "IGR": 'Silent',
                  "Intron": 'Silent',
                  "lincRNA": 'Silent',
                  "RNA": 'Silent',
                  "Silent": 'Silent',
                  "non_coding_transcript_exon": 'Silent',
                  "upstream_gene": 'Silent',
                  "Splice_Region": 'Silent',
                  "Targeted_Region": 'Silent',
                  'Splice_Site': 'LOF',
                  'Nonsense_Mutation': 'LOF',
                  'Frame_Shift_Del': 'LOF',
                  'Frame_Shift_Ins': 'LOF',
                  'Stop_Codon_Del': 'LOF',
                  'Stop_Codon_Ins': 'LOF',
                  'Nonstop_Mutation': 'LOF',
                  'Start_Codon_Del': 'LOF',
                  'Missense_Mutation': 'Other_nonsynonymous',
                  'In_Frame_Del': 'Other_nonsynonymous',
                  'In_Frame_Ins': 'Other_nonsynonymous',
                  'De_novo_Start_InFrame': 'Other_nonsynonymous',
                  'Translation_Start_Site': 'Other_nonsynonymous'}


def load_tcga_dataset(directory_path):
    muts = pd.read_csv(directory_path + '/data_mutations.txt', delimiter='\t')
    grouped_muts = muts[muts['Variant_Classification'].apply(lambda x: MUTATIONS_DICT[x]) != 'Silent'][['Hugo_Symbol',
                                                                                                    'Variant_Classification',
                                                                                                    'Tumor_Sample_Barcode']].groupby(['Tumor_Sample_Barcode',
                                                                                                                                      'Hugo_Symbol']).count()
    rna = pd.read_csv(directory_path + '/data_mrna_seq_v2_rsem.txt',
                      sep='\t',
                      low_memory=False
                      ).dropna().set_index('Hugo_Symbol').drop(['Entrez_Gene_Id'], axis=1).T
    rna = rna.loc[:,~rna.columns.duplicated()].astype(float).copy()
    cna = pd.read_csv(directory_path + '/data_cna.txt',
                      low_memory=False,
                      sep='\t').dropna().set_index('Hugo_Symbol').drop(['Entrez_Gene_Id'], axis=1).T
    cna.drop('Cytoband', errors='ignore', inplace=True)
    cna = cna.loc[:,~cna.columns.duplicated()].astype(float).copy()
    
    genes = list(set(rna.columns).intersection(cna.columns))
    indices = list(set(rna.index).intersection(cna.index))
    tumor_type = pd.DataFrame(len(indices)*[directory_path.split('/')[-1].split('_')[0]],
                              index = indices, columns=['tumor'])
    
    mut = pd.DataFrame(index=rna.index, columns=rna.columns).fillna(0)
    for i in grouped_muts.iterrows():
        try: 
            mut.loc[i[0][0]][i[0][1]] = 1
        except KeyError:
            pass
    return rna[genes], cna[genes], tumor_type, mut


def select_highly_variable_genes(df, bins=10, genes_per_bin=100):
    if len(df.columns) < bins*genes_per_bin:
        raise ValueError('Want to select more genes than present in Input')
    bin_assignment = pd.DataFrame(pd.qcut(df.sum(axis=0), 10, labels=False, duplicates='drop'), columns=['bin'])
    gene_std = pd.DataFrame(df.std(), columns=['std'])
    return bin_assignment.join(gene_std).groupby('bin')['std'].nlargest(genes_per_bin).reset_index()


def draw_auc(fpr, tpr, auc, draw, save=False):
    if isinstance(fpr, list):
        fpr, tpr = fpr[draw], tpr[draw]
    plt.plot(fpr, tpr, color="darkorange", label="ROC curve (area = %0.2f)" % auc)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.legend(loc="lower right")
    if save:
        plt.savefig(save)
    else:
        plt.show()


def get_auc(pred, target, draw=0, save=False):
    if len(target.shape) > 1 and target.shape[1] > 1:
        collapsed_target = target.argmax(axis=1)
    else:
        collapsed_target = target.int()
    num_classes = int(max(collapsed_target) + 1)
    print(pred.shape)
    if pred.shape[1] > 1:
        print('Getting multiclass AUC as One vs. Rest')
        auroc = torchmetrics.AUROC(task='multiclass', num_classes=num_classes)
        roc = torchmetrics.ROC(task='multiclass', num_classes=num_classes)
    else:
        print('doing binary')
        auroc = torchmetrics.AUROC(task='binary')
        roc = torchmetrics.ROC(task='binary')
    auc = auroc(pred, collapsed_target)
    fpr, tpr, tresholds = roc(pred, collapsed_target)
    draw_auc(fpr, tpr, auc, draw, save=save)
    return auc


def select_non_constant_genes(df, cutoff=0.05):
    """
    Many expression datasets have genes that have mostly constant expression throughout the dataset, we can select for
    genes that have minimum percentage of unique values.
    :param df: pd.DataFrame; dataframe to select columns from
    :param cutoff: float; percentage of unique values we require
    :return: list(str); list of genes that are not constant in dataframe
    """
    return list(df.loc[:, df.nunique()/df.count() > 0.05].columns)


def shuffle_connections(mask):
    for i in range(mask.shape[0]):
        np.random.shuffle(mask[i,:])
    return mask


def format_multiclass(target):
    assert len(target.shape) <= 3, '''Three or more dimensional target, I am confused'''
    if len(target.shape) == 1 or target.shape[-1] == 1:
        return make_multiclass_1_hot(target)
    else:
        tensor = torch.tensor(target.values)
        # Verify that each sample is only labelled with one class
        assert torch.allclose(tensor.sum(dim=1), torch.ones(tensor.shape[0])), '''Sum of rows is not equal to one, 
        either some samples have multiple class labels or the target is not one hot encoded.'''
        return target.astype('long')


def make_multiclass_1_hot(target):
    t = torch.tensor(target.values)
    num_classes = int(torch.max(t).item()) + 1
    # Perform one-hot encoding
    binary_labels = F.one_hot(t.view(-1).long(), num_classes)
    # Reshape the binary labels tensor to match the desired shape (N, C)
    binary_labels = binary_labels.view(-1, num_classes)
    return pd.DataFrame(index=target.index, data=binary_labels)


def format_binary(target):
    assert len(target.shape) <= 3, '''Three or more dimensional target, I am confused'''
    if len(target.shape) == 1 or target.shape[-1] == 1:
        assert target.isin([0, 1]).all().all(), '''Binary class labels outside [0, 1] were found'''
        return target.astype('long')
    else:
        tensor = torch.tensor(target.values)
        # Verify that each sample is only labelled with one class
        assert torch.allclose(tensor.sum(dim=1), torch.ones(tensor.shape[0])), '''Sum of rows is not equal to one, 
        either some samples have multiple class labels or the target is not one hot encoded.'''
        positive_label = pd.DataFrame(target).columns[-1]
        target_transformed = (pd.DataFrame(target)[positive_label] == 1).astype('long').to_frame()
        return target_transformed


def format_target(target, task):
    if task == 'MC':
        if target.shape[-1] == 1 or len(target.shape) == 1:
            warnings.warn('''Multiclass labels should be in One-Hot encoded format. Class labels will be coerced
                        this might lead to unintended outcomes''')
        target = format_multiclass(target)
    if task == 'BC':
        target = format_binary(target)
    return target


def get_task(target):
    t = torch.tensor(target.values)
    unique_values = torch.unique(t)
    if len(unique_values) <= 2 and all(value.item() in [0, 1] for value in unique_values):
        if t.shape[-1] == 1 or len(t.shape) == 1:
            # Binary classification
            task_name = 'BC'
        else:
            # Multiclass classification
            task_name = 'MC'
    else:
        # Regression
        task_name = 'REG'
    print('Task defined: {} \n if this is not the intended task please specify task'.format(task_name))
    return task_name


def get_loss_function(task):
    if task == 'BC':
        loss_function = nn.BCEWithLogitsLoss(reduction='sum')
    elif task == 'MC':
        loss_function = nn.CrossEntropyLoss(reduction='sum')
    else:
        loss_function = nn.MSELoss(reduction='sum')
    print('Loss function used: {}'.format(loss_function))
    return loss_function


class EarlyStopper:
    def __init__(self, save_path, patience=1, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.verbose = verbose
        self.save_path = save_path

    def early_stop(self, validation_loss, model):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            torch.save(model.state_dict(), self.save_path)
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_validation_loss*self.min_delta):
            self.counter += 1
            if self.verbose:
                print('exceeded delta')
            if self.counter >= self.patience:
                return True
        return False