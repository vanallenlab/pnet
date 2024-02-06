import networkx as nx
import numpy as np
import pandas as pd


layer_node_list = [2048, 1024, 512, 256, 128, 64, 32, 16]

class GenesetNetwork:
    def __init__(self, gene_list, path, num_layers=3, sparsity=0.9, trim=0):
        # Loading connections and setting up graph
        self.gene_list = gene_list
        self.genes2pathways = self.load_genes2pathways(path)
        self.pathway_encoding = self.load_pathway_encoding()
        
        self.layer_nodes = [self.genes2pathways['pathway'].nunique()] + [l for l in layer_node_list if l<self.genes2pathways['pathway'].nunique()][:num_layers] + [1]
        
        self.gene_layers, self.pathway_layers = self.get_layers(sparsity)
        
#         self.hierarchy = self.load_hierarchy()
#         self.graph = self.generate_graph()

#         # Store metadata and prepare for mask extraction
#         self.max_level = min(self.get_number_of_layers(), max_depth)
#         self.nodes_per_level = self.get_nodes_at_levels()

#         # Generate layers of Graph as Adjacency matrices
#         self.gene_layers, self.pathway_layers = self.get_layers(trim)
#         # Remove gene inputs which flow into children pathways as well
#         for layer in reversed(self.gene_layers[1:]):
#             for pathway in layer.columns:
#                 self.clean_redundant_gene_input(pathway)
                
                
    def load_genes2pathways(self, path):
        """
        Loads the gene to pathway edges from the gmt file. Produces a dataframe with the pathway code and the single
        HUGO gene IDs as columns. Contains an entry for every relation in the Reactome dataset. Adding a residual node
        for genes in the dataset which have no connection to Reactome.
        :return: DataFrame with columns ['pathway', 'gene'] with binary relations between pathways and genes
        """
        filename = path
        genes_start_col = 2
        pathway_col = 0
        genes2pathways_list = []
        with open(filename) as gmt:
            lines = gmt.readlines()
            for line in lines:
                line_list = line.strip().split('\t')
                pathway = line_list[pathway_col]
                for gene in line_list[genes_start_col:]:
                    if gene in self.gene_list:
                        genes2pathways_list.append({'pathway': pathway, 'gene': gene})
        genes2pathways = pd.DataFrame(genes2pathways_list)

        # connect unused genes to a residual node in the last layer
        unused_genes = list(set(self.gene_list).difference(set(genes2pathways['gene'].unique())))
        unused_genes_df = pd.DataFrame(columns=['pathway', 'gene'])
        unused_genes_df['gene'] = unused_genes
        unused_genes_df['pathway'] = 'residual'
        return pd.concat([genes2pathways, unused_genes_df]).reset_index(drop=True)
    
    
    def load_pathway_encoding(self):
        """
        Placeholder function to keep structure of Reactome Network, not needed for Gene Sets, but keeps functionality of network interpretation
        """
        data1 = {'ID': self.genes2pathways['pathway'].unique(),
                'pathway': self.genes2pathways['pathway'].unique(),
                'species': len(self.genes2pathways['pathway'].unique())*['Homo sapiens']
               }
        
        data2 = {'ID': ['rand_'+str(x) for x in range(2048)], 
                 'pathway': ['rand_'+str(x) for x in range(2048)],
                 'species': ['Homo sapiens' for x in range(2048)]}
        return pd.concat([pd.DataFrame(data1), pd.DataFrame(data2)]).reset_index(drop=True)
    
    
    def get_layers(self, sparsity):
        gene_layers = []
        first_gene_layer = pd.get_dummies(self.genes2pathways['gene']).join(self.genes2pathways['pathway']).groupby('pathway').sum().T
        gene_layers.append(first_gene_layer)
        for i in range(len(self.layer_nodes)-2):
            rand_gene_layer = np.random.choice([0, 1], size=(len(self.gene_list), self.layer_nodes[i+1]), p=[sparsity, 1-sparsity])
            layer_columns = ['rand_'+str(c) for c in range(rand_gene_layer.shape[1])]
            layer_index = ['rand_'+str(c) for c in range(rand_gene_layer.shape[0])]
            gene_layers.append(pd.DataFrame(rand_gene_layer, index=layer_index, columns=layer_columns))
            
        pathway_layers = []
        first_pathway_layer = np.random.choice([0, 1], size=(self.layer_nodes[0],self.layer_nodes[1]), p=[sparsity, 1-sparsity])
        layer_columns = ['rand_'+str(c) for c in range(first_pathway_layer.shape[1])]
        pathway_layers.append(pd.DataFrame(first_pathway_layer, index=first_gene_layer.columns, columns=layer_columns))
        for i in range(1, len(self.layer_nodes)-1):
            rand_pathway_layer = np.random.choice([0, 1], size=(self.layer_nodes[i],self.layer_nodes[i+1]), p=[sparsity, 1-sparsity])
            layer_columns = ['rand_'+str(c) for c in range(rand_pathway_layer.shape[1])]
            layer_index = ['rand_'+str(c) for c in range(rand_pathway_layer.shape[0])]
            pathway_layers.append(pd.DataFrame(rand_pathway_layer, columns=layer_columns, index=layer_index))
        return gene_layers, pathway_layers
    
    
    def get_masks(self, nbr_genetic_input_types):
        """
        Transforms pd.DataFrame adjacency matrices into binary np.array masks. Input layer connections based on the
         number of genetic inputs.
        :param nbr_genetic_input_types: int; number of genetic input modalities provided
        :return: (list(np.array), list(np.array), np.array); a list of adjacency matrices per layer and a list of
            gene-to-pathway adjacency matrix per layer. The input mask to connect the same gene from different
             modalities to the input node
        """
        input_mask = pd.DataFrame(index=nbr_genetic_input_types*self.gene_list, columns=self.gene_list).fillna(0)
        for col in input_mask.columns:
            input_mask[col].loc[col] = 1
        gene_masks = [l.values for l in self.gene_layers]
        pathway_masks = [l.values for l in self.pathway_layers]
        return gene_masks, pathway_masks, input_mask.values
    
    
    def are_there_bugs(self):
        # FIXME; this should return True when there are bugs.
        there_are_no_bugs = True
        if there_are_no_bugs:
            return False
        else:
            return True


