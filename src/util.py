import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchmetrics

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


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.verbose = verbose

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_validation_loss*self.min_delta):
            self.counter += 1
            if self.verbose:
                print('exceeded delta')
            if self.counter >= self.patience:
                return True
        return False
