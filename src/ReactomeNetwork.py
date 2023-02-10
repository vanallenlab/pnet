import networkx as nx
import numpy as np
import pandas as pd


class ReactomeNetwork:
    def __init__(self, gene_list, trim=0):
        # Loading connections and setting up graph
        self.gene_list = gene_list
        self.pathway2genes = self.load_pathway2genes()
        self.pathway_encoding = self.load_pathway_encoding()
        self.hierarchy = self.load_hierarchy()
        self.graph = self.generate_graph()

        # Store metadata and prepare for mask extraction
        self.max_level = self.get_number_of_layers()
        self.nodes_per_level = self.get_nodes_at_levels()

        # Generate layers of Graph as Adjacency matrices
        self.gene_layers, self.pathway_layers = self.get_layers(trim)

    def load_pathway2genes(self):
        """
        Loads the gene to pathway edges from the gmt file. Produces a dataframe with the pathway code and the single
        HUGO gene IDs as columns. Contains an entry for every relation in the Reactome dataset.
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
        return pathway2genes

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
        root node is equivalent to the prediction head.
        :return: networkX graph of reactome network
        """
        highest_level_pathways = self.hierarchy[~self.hierarchy['source'].isin(self.hierarchy['target']
                                                                               .unique())]['source'].unique()
        G = nx.from_pandas_edgelist(self.hierarchy, 'source', 'target', create_using=nx.DiGraph())
        G.add_node('root')
        for pathway in highest_level_pathways:
            G.add_edge('root', pathway)
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

    def get_layers(self, trim):
        """
        Generating a pd.DataFrame with the adjacency matrix between nodes of one layer to the next. An adjacency matrix
        for each level is generated. Additionally, an adjacency matrix for genes to each layer is generated. This second
        adjacency matrix connects genes directly to higher level pathways.
        :param trim: int; number of minimum inflows to a node to keep the node in the network.
        :return: (list(pd.DataFrame), list(pd.DataFrame)); a list of adjacency matrices per layer and a list of
            gene-to-pathway adjacency matrix per layer.
        """
        gene_layers = []
        pathway_layers = []
        for level in reversed(range(self.max_level)):
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

    def get_masks(self, nbr_genetic_input_types):
        """
        Transforms pd.DataFrame adjacency matrices into binary np.array masks. Multiplying masks for gene level
         connections based on the number of genetic inputs.
        :param nbr_genetic_input_types: int; number of genetic input modalities provided
        :return: (list(np.array), list(np.array)); a list of adjacency matrices per layer and a list of
            gene-to-pathway adjacency matrix per layer.
        """
        gene_masks = [np.concatenate(nbr_genetic_input_types*[l.values]) for l in self.gene_layers]
        pathway_masks = [l.values for l in self.pathway_layers]
        return gene_masks, pathway_masks


