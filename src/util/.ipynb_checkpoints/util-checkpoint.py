import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import torchmetrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, auc


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
                  'De_novo_Start_OutOfFrame': 'Other_nonsynonymous',
                  'Translation_Start_Site': 'Other_nonsynonymous',
                  'Start_Codon_SNP': 'Other_nonsynonymous',
                  'Start_Codon_Ins': 'LOF'}


def load_tcga_dataset(directory_path, load_mut=False, rna_standardized=True):
    if rna_standardized:
        rna = pd.read_csv(directory_path + '/data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt',
                          sep='\t',
                          low_memory=False
                          ).dropna().set_index('Hugo_Symbol').drop(['Entrez_Gene_Id'], axis=1).T
    else:
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
    
    if load_mut:
        muts = pd.read_csv(directory_path + '/data_mutations.txt', delimiter='\t')
        grouped_muts = muts[muts['Variant_Classification'].apply(lambda x: MUTATIONS_DICT[x]) != 'Silent'][['Hugo_Symbol',
                                                                                                        'Variant_Classification',
                                                                                                        'Tumor_Sample_Barcode']].groupby(['Tumor_Sample_Barcode',
                                                                                                                                          'Hugo_Symbol']).count()
        mut = grouped_muts.unstack(level=-1).fillna(0).droplevel(0, axis=1)
        
        genes = list(set(genes).intersection(mut.columns))
        indices = list(set(indices).intersection(mut.index))
        return rna[genes], cna[genes], tumor_type, mut[genes]
    else:
        return rna[genes], cna[genes], tumor_type


def select_highly_variable_genes(df, bins=10, genes_per_bin=100):
    if len(df.columns) < bins*genes_per_bin:
        raise ValueError('Want to select more genes than present in Input')
    bin_assignment = pd.DataFrame(pd.qcut(df.sum(axis=0), 10, labels=False, duplicates='drop'), columns=['bin'])
    gene_std = pd.DataFrame(df.std(), columns=['std'])
    return bin_assignment.join(gene_std).groupby('bin')['std'].nlargest(genes_per_bin).reset_index()


def draw_auc(fpr, tpr, auc_score, draw, save=False):
    if isinstance(fpr, list):
        fpr, tpr = fpr[draw], tpr[draw]
    plt.plot(fpr, tpr, color="darkorange", label="ROC curve (area = %0.2f)" % auc_score)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.legend(loc="lower right")
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    if save:
        plt.savefig(save)
    else:
        plt.show()
        
def draw_loss(train_scores, test_scores, save=False):
    epochs = range(1, len(train_scores) + 1)

    plt.plot(epochs, train_scores, label='Train Loss', color='navy')
    plt.plot(epochs, test_scores, label='Test Loss', color='indianred')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss Curves')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend()
    if save:
        plt.savefig(save)
    else:
        plt.show()


def get_auc(pred_proba, target, draw=0, save=False):
    target=target.to(torch.int)
    if len(target.shape) > 1 and target.shape[1] > 1:
        auc_score = multiclass_auc(pred_proba, target, save)
    else:
        collapsed_target = target.int()
        auroc = torchmetrics.AUROC(task='binary')
        roc = torchmetrics.ROC(task='binary')
        auc_score = auroc(pred_proba, collapsed_target)
        fpr, tpr, tresholds = roc(pred_proba, collapsed_target)
        draw_auc(fpr, tpr, auc_score, draw, save=save)
    return auc_score


def get_auc_prc(pred_proba, target):
    target=target.to(torch.int)
    if len(target.shape) > 1 and target.shape[1] > 1:
        num_classes = target.shape[1]
        target = target.argmax(axis=1)
        auc_prc = torchmetrics.functional.average_precision(pred_proba, target, task='multiclass', num_classes=num_classes)
    else:
        auc_prc = torchmetrics.functional.average_precision(pred_proba, target, task='binary')
    return auc_prc.item() 


def get_f1(pred, target):
    target=target.to(torch.int)
    pred=pred.to(torch.int)
    if len(target.shape) > 1 and target.shape[1] > 1:
        num_classes = target.shape[1]
        f1_scores = torchmetrics.functional.f1_score(pred, target, num_classes=num_classes, task='multiclass')
    else:
        f1_scores = torchmetrics.functional.f1_score(pred, target, task='binary')
    return f1_scores


def multiclass_auc(pred_proba, target, save=False):
    # Get the predicted class labels from the probabilities
    predicted_labels = np.argmax(pred_proba, axis=1)

    # Calculate the AUC and ROC curves for each class
    num_classes = pred_proba.shape[1]
    auc_scores = []
    roc_curves = []

    for i in range(num_classes):
        y_true = target[:, i]
        y_score = pred_proba[:, i]

        # Calculate AUC
        auc_score = roc_auc_score(y_true, y_score)
        auc_scores.append(auc_score)

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_curves.append((fpr, tpr))

    for i in range(num_classes):
        fpr, tpr = roc_curves[i]
        roc_auc = auc(fpr, tpr)
        draw_auc(fpr, tpr, roc_auc, draw=0, save=save)
    return auc_scores


def select_non_constant_genes(df, cutoff=0.05):
    """
    Many expression datasets have genes that have mostly constant expression throughout the dataset, we can select for
    genes that have minimum percentage of unique values.
    :param df: pd.DataFrame; dataframe to select columns from
    :param cutoff: float; percentage of unique values we require
    :return: list(str); list of genes that are not constant in dataframe
    """
    return df.loc[:, df.nunique()/df.count() > 0.05]


def shuffle_connections(mask):
    for i in range(mask.shape[0]):
        np.random.shuffle(mask[i,:])
    return mask


def format_multiclass(target):
    assert len(target.shape) <= 3, '''Three or more dimensional target, I am confused'''
    if len(target.shape) == 1 or target.shape[-1] == 1:
        return make_multiclass_1_hot(target)
    else:
        tensor = torch.tensor(target.values, dtype=torch.float)
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


def get_class_weights(target):
    class_counts = torch.bincount(target)
    total_samples = class_counts.sum().float()

    # Calculate inverse class frequencies
    class_weights = total_samples / (class_counts.float() + 1e-7)  # Add small epsilon to avoid division by zero

    # Normalize class weights
    class_weights = class_weights / class_weights.sum()
    return class_weights


def BCELoss_class_weighted(weights):

    def loss(input, target):
        input = torch.clamp(input,min=1e-7,max=1-1e-7)
        bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
        return torch.sum(bce)
    return loss


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