import networkx as nx
import numpy as np
import pandas as pd


class ReactomeNetwork:
    def __init__(self, gene_list, trim=0, max_depth=6, pathways_to_drop=[]):
        # Loading connections and setting up graph
        self.gene_list = gene_list
        self.pathway2genes = self.load_pathway2genes()
        self.pathway_encoding = self.load_pathway_encoding()
        self.hierarchy = self.load_hierarchy()
        self.graph = self.generate_graph()
        self.drop_pathways(pathways_to_drop)
        self.reg_relations= pd.read_csv('../data/regulatory/collectri_filtered.csv')

        # Store metadata and prepare for mask extraction
        self.max_level = min(self.get_number_of_layers(), max_depth)
        self.nodes_per_level = self.get_nodes_at_levels()

        # Generate layers of Graph as Adjacency matrices
        self.gene_layers, self.pathway_layers = self.get_layers(trim)
        # Remove gene inputs which flow into children pathways as well
        for layer in reversed(self.gene_layers[1:]):
            for pathway in layer.columns:
                self.clean_redundant_gene_input(pathway)

    def load_pathway2genes(self):
        """
        Loads the gene to pathway edges from the gmt file. Produces a dataframe with the pathway code and the single
        HUGO gene IDs as columns. Contains an entry for every relation in the Reactome dataset. Adding a residual node
        for genes in the dataset which have no connection to Reactome.
        :return: DataFrame with columns ['pathway', 'gene'] with binary relations between pathways and genes
        """
        filename = '../data/reactome/ReactomePathways.gmt'
        genes_start_col = 2
        pathway_col = 1
        pathway2genes_list = []
        with open(filename) as gmt:
            lines = gmt.readlines()
            for line in lines:
                line_list = line.strip().split('\t')
                pathway = line_list[pathway_col]
                for gene in line_list[genes_start_col:]:
                    if gene in self.gene_list:
                        pathway2genes_list.append({'pathway': pathway, 'gene': gene})
        pathway2genes = pd.DataFrame(pathway2genes_list)

        # connect unused genes to a residual node in the last layer
        unused_genes = list(set(self.gene_list).difference(set(pathway2genes['gene'].unique())))
        unused_genes_df = pd.DataFrame(columns=['pathway', 'gene'])
        unused_genes_df['gene'] = unused_genes
        unused_genes_df['pathway'] = 'residual'
        return pd.concat([pathway2genes, unused_genes_df]).reset_index(drop=True)

    @staticmethod
    def load_pathway_encoding(species="HSA"):
        """
        Generates an encoding DataFrame for the pathway codes filtered for a given species
        :param species: string of species to filter pathways for, default is HSA for human pathways
        :return: DataFrame with columns ['ID','pathway','species']
        """
        filename = '../data/reactome/ReactomePathways.txt'
        df = pd.read_csv(filename, sep="\t")
        df.columns = ["ID", "pathway", "species"]
        df = df[df["ID"].str.contains(species)]
        df.loc[len(df)] = ['residual', 'residual', 'Homo sapiens']      # Adding residual node for completion
        return df

    @staticmethod
    def load_hierarchy(species="HSA"):
        """
        Generates a parent-child DataFrame for all pathway dependencies in the Reactome network. Filters for the given
        species
        :param species: string of species to filter for, default is HSA for human
        :return: DataFrame with columns ['source','target'] for each parent-child relation in Reactome
        """
        filename = '../data/reactome/ReactomePathwaysRelation.txt'
        df = pd.read_csv(filename, sep="\t")
        df.columns = ["source", "target"]
        df = df[df["target"].str.contains(species)]
        return df

    def generate_graph(self):
        """
        Generates networkX graph from hierarchy edge list. Connecting all highest order pathways to a root node. The
        root node is equivalent to the prediction head. Adding a connection of the residual (unconnected genes) to the
        root node.
        :return: networkX graph of reactome network
        """
        highest_level_pathways = self.hierarchy[~self.hierarchy['source'].isin(self.hierarchy['target']
                                                                               .unique())]['source'].unique()
        G = nx.from_pandas_edgelist(self.hierarchy, 'source', 'target', create_using=nx.DiGraph())
        G.add_node('root')
        for pathway in highest_level_pathways:
            G.add_edge('root', pathway)
        G.add_node('residual')
        G.add_edge('root', 'residual')
        return G

    def get_nodes_at_level(self, level):
        """
        returns all the nodes that are on given pathway level
        :param graph: nx graph containing all the pathways
        :param level: int level to get nodes from
        :return: list of nodes on the given level
        """
        nodes = set(nx.ego_graph(self.graph, 'root', radius=level))
        if level >= 1.:         # remove nodes that are not **at** the specified distance but closer
            nodes -= set(nx.ego_graph(self.graph, 'root', radius=level - 1))
        return list(nodes)

    def get_pathway_level(self, pathway):
        """
        :param pathway: str; code of the pathway
        :return: int; level of the pathway in the network layers
        """
        for i, layer in enumerate(self.gene_layers):
            if pathway in layer.columns:
                return i
        raise KeyError("Pathway {} not found".format(pathway))

    def get_children_gene_inputs(self, level, pathway):
        """
        Generates a list of genes that flow as input into all children of the pathway.
        :param level: int; pathway level in the network
        :param pathway: str; name of the pathway
        :return: List(str); all genes flowing into children pathways
        """
        p_adjacency = self.pathway_layers[level - 1]
        g_adjacency = self.gene_layers[level - 1]
        children = list(p_adjacency[p_adjacency[pathway] == 1].index)
        return list(g_adjacency[g_adjacency[children].sum(axis=1) > 0].index)

    def clean_redundant_gene_input(self, pathway):
        """
        Hierarchical structure of reactome connects all children gene inputs to parent node as well. We want these gene
        connections only to flow through the respective pathways. Therefore, we need to remove all gene connections
        which are connected to children of a pathway.
        :param pathway: str; name of the pathway
        :return: void; setting the respective gene_layer connections to 0
        """
        level = self.get_pathway_level(pathway)
        children_genes = self.get_children_gene_inputs(level, pathway)
        self.gene_layers[level][pathway][children_genes] = 0

    def get_nodes_at_levels(self):
        """
        :return: list(list(nodes)) a list containing the list of nodes per level, from lowest to highest level.
        """
        nodes_per_level = []
        for i in range(self.max_level):
            nodes_per_level.append(self.get_nodes_at_level(i))
        return nodes_per_level

    def get_number_of_layers(self):
        return nx.dag_longest_path_length(self.graph)

    def get_number_of_inputs(self, node):
        """
        Counting inflow edges of a given node
        :param node: networkX.node; node on which inflow should be counted
        :return: int number of in edges to node
        """
        input_pathways = [n[1] for n in self.graph.out_edges(node)]
        input_genes = list(self.pathway2genes[self.pathway2genes['pathway'] == node]['gene'])
        return len(input_genes + input_pathways)

    
    def drop_pathways(self, pathways=[]):
        id_pathways = self.pathway_encoding[self.pathway_encoding['ID'].isin(pathways)]
        name_pathways = self.pathway_encoding[self.pathway_encoding['pathway'].isin(pathways)]
        to_drop = pd.concat([id_pathways, name_pathways]).drop_duplicates('ID')
        
        for i in to_drop['ID']:
            self.graph.remove_node(i)
            print('removed: ', to_drop[to_drop['ID'] == i]['pathway'].values[0])
        
    def get_layers(self, trim):
        """
        Generating a pd.DataFrame with the adjacency matrix between nodes of one layer to the next. An adjacency matrix
        for each level is generated. Additionally, an adjacency matrix for genes to each layer is generated. This second
        adjacency matrix connects genes directly to higher level pathways.
        :param trim: int; number of minimum inflows to a node to keep the node in the network.
        :param depth: int: number of pathway levels to be considered for the network.
        :return: (list(pd.DataFrame), list(pd.DataFrame)); a list of adjacency matrices per layer and a list of
            gene-to-pathway adjacency matrix per layer.
        """
        gene_layers = []
        pathway_layers = []
        for level in reversed(range(1, self.max_level)):
            pathway_nodes = self.nodes_per_level[level]
            higher_level_pathway_nodes = self.nodes_per_level[level-1]
            if pathway_nodes:       # Only add layers if there is nodes in layer
                # Generate empty adjacency matrices for each layer
                gene_connections = pd.DataFrame(index=self.gene_list, columns=pathway_nodes).fillna(0)
                pathway_connections = pd.DataFrame(index=pathway_nodes, columns=higher_level_pathway_nodes).fillna(0)
                for pathway in pathway_nodes:
                    if self.get_number_of_inputs(pathway) > trim:
                        # Add connections only if there are sufficient inflows
                        # Add genes to pathways connections to adjacency of layer
                        genes_in_pathway = self.pathway2genes[self.pathway2genes['pathway'] == pathway]['gene']
                        gene_connections[pathway][genes_in_pathway] = 1

                        # Add pathway to pathways connections to adjacency of layer
                        pathways_in_pathway = [n[0] for n in self.graph.in_edges(pathway)]
                        pathways_in_pathway = list(set(pathways_in_pathway).intersection(higher_level_pathway_nodes))
                        for p in pathways_in_pathway:
                            pathway_connections[p][pathway] = 1

                gene_layers.append(gene_connections)
                pathway_layers.append(pathway_connections)
        return gene_layers, pathway_layers
    
    def get_reg_mask(self):
        reg_relations_filtered = self.reg_relations.loc[self.reg_relations['Origin'].isin(self.gene_list) &self.reg_relations['Target'].isin(self.gene_list)]
        reg_origins = set(reg_relations_filtered['Origin'].values)
        extra_mask = pd.DataFrame(index=self.gene_list, columns=self.gene_list).fillna(0)
        
        for col in reg_origins:
            extra_mask[col].loc[col]=1

            matched_indices = reg_relations_filtered.loc[reg_relations_filtered['Origin'] == col, 'Target'].values
            for ind in matched_indices:
                extra_mask[col].loc[ind]= reg_relations_filtered['weight'].loc[(reg_relations_filtered['Origin']==col) &(reg_relations_filtered['Target']==ind)].values 
        print('Added regulatory layer')
        return extra_mask
    

    def get_masks(self, nbr_genetic_input_types, regulatory=False):
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
        if regulatory:
            reg_mask = self.get_reg_mask()
            return gene_masks, pathway_masks, input_mask.values, reg_mask.values
        else:
            return gene_masks, pathway_masks, input_mask.values