import os
import pandas as pd
import numpy as np
from pnet import ReactomeNetwork
import plotly.graph_objects as go
from scipy.stats import zscore

GENE_COLOR = '#41B6E6'
PATHWAY_COLOR = '#00629B'
RES_COLOR = '#FFA300'

class SankeyDiag:
    def __init__(self, imps_dir, target=None, runs=1):
        if runs>1:
            self.all_imps = self.load_multiple_runs(imps_dir, runs)
        else:
            self.all_imps = self.load_importance_scores(imps_dir)
        self.grouped_imps = self.group_importance_scores(target)
        self.normalize_layerwise()
        self.rn = self.get_reactome_network_for_imps()
        self.nbr_displayed = 5

        self.initialize_links()
        self.gene_feature_to_gene_layer()
        self.gene_to_pathway_layer()
        for i in range(1, 5):
            self.add_pathway_layer_to_sankey(i)
        self.pathway_to_output_layer()
        self.short_name_dict = self.get_short_names()
        
        
    def load_multiple_runs(self, results_dir, runs):
        all_importances = pd.DataFrame()
        for i in range(runs):
            run_i_imps = self.load_importance_scores('{}/run{}/'.format(results_dir,i))
            run_i_imps['Run'] = i
            all_importances = pd.concat([all_importances, run_i_imps])
        return all_importances
    
        
    def load_importance_scores(self, results_dir):
        layer_list = [x for x in os.listdir(results_dir) if x[-3:] == 'csv' and x[:5] == 'layer']
        layer_list = ['gene_feature', 'gene'] + sorted(['_'.join(x.split('_')[:-1]) for x in layer_list])

        all_imps = pd.DataFrame(columns=['Importance', 'Layer'])
        for l in layer_list:
            df_imps = pd.DataFrame(columns=['Importance', 'Layer'])
            imps = pd.read_csv('{}{}_importances.csv'.format(results_dir, l)).set_index('Unnamed: 0')
            df_imps = imps.reset_index().melt(id_vars='Unnamed: 0', var_name='Gene/Pathway', value_name='Importance').rename(columns={'Unnamed: 0': 'Sample'})
            df_imps['Layer'] = l
            all_imps = pd.concat([all_imps, df_imps])
        return all_imps


    def group_importance_scores(self, target):
        if target is not None:
            response = target.columns[0]
            imps_w_target = self.all_imps.merge(target, left_on='Sample', right_index=True)
            grouped_imps = imps_w_target.groupby(['Gene/Pathway', 'Layer', response]).mean().diff().abs()
            grouped_imps = grouped_imps.query('{} == 1'.format(response)).reset_index()
        else:
            grouped_imps = pd.DataFrame(self.all_imps.groupby(['Gene/Pathway', 'Layer']).mean().abs().reset_index())
        return grouped_imps
    
    
    def normalize_layerwise(self):
        layer_normalized_imps = pd.Series(dtype=float)
        for l in self.grouped_imps['Layer'].unique():
            normalized = NormalizeData(self.grouped_imps[self.grouped_imps['Layer']==l]['Importance'])
            layer_normalized_imps = layer_normalized_imps.append(normalized)
        self.grouped_imps['Importance'] = layer_normalized_imps

        
    def get_reactome_network_for_imps(self):
        self.genes = self.grouped_imps[self.grouped_imps['Layer']=='gene']['Gene/Pathway'].unique()
        rn = ReactomeNetwork.ReactomeNetwork(self.genes)
        pathway_encoding = rn.pathway_encoding.set_index('ID')['pathway'].to_dict()
        rn.pathway2genes['pathway'] = rn.pathway2genes['pathway'].apply(lambda x: pathway_encoding[x])
        rn.hierarchy['source'] = rn.hierarchy['source'].apply(lambda x: pathway_encoding[x]) 
        rn.hierarchy['target'] = rn.hierarchy['target'].apply(lambda x: pathway_encoding[x])
        return rn


    def append_links(self, source, target, value, color):
        self.links['source'].append(source)
        self.links['target'].append(target)
        self.links['value'].append(value)
        self.links['color'].append(color)


    def initialize_links(self):
        self.links = dict(source = [], target = [], value = [], color=[])
        self.numerical_encoding = {}
        self.num = 0
        
    
    def add_to_num_encoding(self, elem):
        if elem not in self.numerical_encoding:
                self.numerical_encoding[elem] = self.num
                self.num+=1


    def gene_feature_to_gene_layer(self):
        gene_feature_imps =self.grouped_imps[self.grouped_imps['Layer'] == 'gene_feature'].copy()
        gene_imps = self.grouped_imps[self.grouped_imps['Layer'] == 'gene'].copy()
        self.add_to_num_encoding('Residual_0')
        self.add_to_num_encoding('Residual_1')

        # Add Genes that have inflow from specific features
        for ind, elem in gene_feature_imps.nlargest(self.nbr_displayed, 'Importance').iterrows():
            source = elem['Gene/Pathway']
            target = elem['Gene/Pathway'].split('_')[0]
            value = elem['Importance']

            self.add_to_num_encoding(source)
            self.add_to_num_encoding(target)

            # Change target to Residual when not important enough  
            if target not in list(gene_imps.nlargest(self.nbr_displayed, 'Importance')['Gene/Pathway']):
                target = 'Residual_1'
                self.append_links(self.numerical_encoding[source], self.numerical_encoding[target], value, RES_COLOR)
            else:
                self.append_links(self.numerical_encoding[source], self.numerical_encoding[target], value, GENE_COLOR)

        # Add Genes that have only inflow from Residual
        for ind, elem in gene_imps.nlargest(self.nbr_displayed, 'Importance').iterrows():
            target = elem['Gene/Pathway']
            source = 'Residual_0'
            value = gene_feature_imps.nsmallest(gene_feature_imps.shape[0]-self.nbr_displayed, 'Importance')['Importance'].mean()
            self.add_to_num_encoding(target)

            if self.numerical_encoding[target] not in self.links['target']:
                self.append_links(self.numerical_encoding[source], self.numerical_encoding[target], value, RES_COLOR)

        # Add Residual connection
        target = 'Residual_1'
        source = 'Residual_0'
        value = gene_imps.nsmallest(gene_feature_imps.shape[0]-self.nbr_displayed, 'Importance')['Importance'].mean()
        self.append_links(self.numerical_encoding[source], self.numerical_encoding[target], value, RES_COLOR)


    def gene_to_pathway_layer(self):
        gene_imps = self.grouped_imps[self.grouped_imps['Layer'] == 'gene'].copy()
        pathway_imps = self.grouped_imps[self.grouped_imps['Layer'] == 'layer_0'].copy()
        #pathway_imps = normalize_imps(pathway_imps)
        self.add_to_num_encoding('Residual_2')

        # Add Pathways that have inflow from specific Genes
        for g_ind, g_elem in gene_imps.nlargest(self.nbr_displayed, 'Importance').iterrows():
            source = g_elem['Gene/Pathway']
            value = g_elem['Importance']
            targets = set()
            for p_ind, p_elem in pathway_imps.nlargest(self.nbr_displayed, 'Importance').iterrows():
                target = p_elem['Gene/Pathway']

                self.add_to_num_encoding(source)
                self.add_to_num_encoding(target)
                # Change target to Residual when no important enough connections
                if target in list(self.rn.pathway2genes[self.rn.pathway2genes['gene']==source]['pathway']):
                    targets.add(target)
            if len(targets) == 0:
                targets.add('Residual_2')
            for target in targets:
                col = RES_COLOR if target == 'Residual_2' else GENE_COLOR
                self.append_links(self.numerical_encoding[source], self.numerical_encoding[target], value, col)

        # Add Pathways that have only inflow from Residual
        for p_ind, p_elem in pathway_imps.nlargest(self.nbr_displayed, 'Importance').iterrows():
            target = p_elem['Gene/Pathway']
            source = 'Residual_1'
            value = gene_imps.nsmallest(gene_imps.shape[0]-self.nbr_displayed, 'Importance')['Importance'].mean()
            self.add_to_num_encoding(target)

            if self.numerical_encoding[target] not in self.links['target']:
                self.append_links(self.numerical_encoding[source], self.numerical_encoding[target], value, RES_COLOR)

        # Add Residual connection
        target = 'Residual_2'
        source = 'Residual_1'
        value = gene_imps.nsmallest(gene_imps.shape[0]-self.nbr_displayed, 'Importance')['Importance'].mean()
        self.append_links(self.numerical_encoding[source], self.numerical_encoding[target], value, RES_COLOR)


    def add_pathway_layer_to_sankey(self, num_layer):
        pathway_higher_imps = self.grouped_imps[self.grouped_imps['Layer']=='layer_{}'.format(num_layer)].copy()
        #pathway_higher_imps = normalize_imps(pathway_higher_imps)
        pathway_lower_imps = self.grouped_imps[self.grouped_imps['Layer']=='layer_{}'.format(num_layer-1)].copy()
        #pathway_lower_imps = normalize_imps(pathway_lower_imps)
        self.add_to_num_encoding('Residual_{}'.format(num_layer+2))

        # Add Pathways that have inflow from specific Pathways
        for p0_ind, p0_elem in pathway_lower_imps.nlargest(self.nbr_displayed, 'Importance').iterrows():
            source = p0_elem['Gene/Pathway']
            value = p0_elem['Importance']
            targets = set()
            for p1_ind, p1_elem in pathway_higher_imps.nlargest(self.nbr_displayed, 'Importance').iterrows():
                target = p1_elem['Gene/Pathway']
                
                self.add_to_num_encoding(source)
                self.add_to_num_encoding(target)

                # Change target to Residual when no important enough connections
                if target in list(self.rn.hierarchy[self.rn.hierarchy['target']==source]['source']):
                    targets.add(target)
            if len(targets) == 0:
                targets.add('Residual_{}'.format(num_layer+2))
            for target in targets:
                col = RES_COLOR if target == 'Residual_{}'.format(num_layer+2) else PATHWAY_COLOR
                self.append_links(self.numerical_encoding[source], self.numerical_encoding[target], value, col)

        # Add Pathways that have only inflow from Residual
        for p_ind, p_elem in pathway_higher_imps.nlargest(self.nbr_displayed, 'Importance').iterrows():
            target = p_elem['Gene/Pathway']
            source = 'Residual_{}'.format(num_layer+1)
            value = pathway_lower_imps.nsmallest(pathway_lower_imps.shape[0]-self.nbr_displayed, 'Importance')['Importance'].mean()
            self.add_to_num_encoding(target)
            if self.numerical_encoding[target] not in self.links['target']:
                self.append_links(self.numerical_encoding[source], self.numerical_encoding[target], value, RES_COLOR)

        # Add Residual connection
        target = 'Residual_{}'.format(num_layer+2)
        source = 'Residual_{}'.format(num_layer+1)
        value = pathway_lower_imps.nsmallest(pathway_lower_imps.shape[0]-self.nbr_displayed, 'Importance')['Importance'].mean()
        self.append_links(self.numerical_encoding[source], self.numerical_encoding[target], value, RES_COLOR)


    def pathway_to_output_layer(self):
        pathway_imps = self.grouped_imps[self.grouped_imps['Layer'] == 'layer_4'].copy()
        #pathway_imps = normalize_imps(pathway_imps)
        self.add_to_num_encoding('Output')

        # Add Pathways that have inflow from specific Genes
        for p_ind, p_elem in pathway_imps.nlargest(self.nbr_displayed, 'Importance').iterrows():
            source = p_elem['Gene/Pathway']
            value = p_elem['Importance']
            target = 'Output'

            col = PATHWAY_COLOR
            self.append_links(self.numerical_encoding[source], self.numerical_encoding[target], value, col)

        self.append_links(self.numerical_encoding['Residual_6'], self.numerical_encoding[target], value, RES_COLOR)


    def get_short_names(self):
        short_names = pd.read_csv('/mnt/disks/pancan/data/pathways_short_names.csv')
        short_names['Short name (Eli)'].fillna(short_names['Short name (automatic)'], inplace=True)
        short_names.set_index('Full name', inplace=True)
        short_name_dict = short_names['Short name (Eli)'].to_dict()
        for i in range(7):
            short_name_dict['Residual_{}'.format(i)] = 'Residual'
        for elem in self.numerical_encoding:
            short_name_dict[elem] = short_name_dict.get(elem, elem)
        return short_name_dict


    def get_sankey_diag(self, savepath):
        fig = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15,
              thickness = 10,
              line = dict(color = "black", width = 0.5),
              label = list([self.short_name_dict[x] for x in self.numerical_encoding.keys()]),
              color = 'silver'
            ),
            link = self.links)])

        fig.update_layout(font_size=10)
        fig.write_html(savepath)
        fig.show()
        return fig
    
    
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))