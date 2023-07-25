import networkx as nx
import numpy as np
import pandas as pd
import re

gene2pathway_loc = '../data/reactome/ReactomePathways.gmt'
hierarchy = filename = '../data/reactome/ReactomePathwaysRelation.txt'
pathway_encoding = '../data/reactome/ReactomePathways.txt'


def load_genes(filename=gene2pathway_loc, genes_start_col=2, pathway_col=1):
    data_dict_list = []
    with open(filename) as gmt:

        data_list = gmt.readlines()

        for row in data_list:
            genes = row.strip().split('\t')
            # genes = [re.sub('_copy.*', '', g) for g in genes]
            # genes = [re.sub('\\n.*', '', g) for g in genes]  ## why??????????
            for gene in genes[genes_start_col:]:
                pathway = genes[pathway_col]
                dict = {'pathway': pathway, 'gene': gene}
                data_dict_list.append(dict)

    df = pd.DataFrame(data_dict_list)

    return df


def load_pathway_encoding(filename=pathway_encoding, species="HSA"):
    df = pd.read_csv(filename, sep="\t")
    df.columns = ["ID", "pathway", "species"]
    df = df[df["ID"].str.contains(species)]
    return df


def load_hierarchy(filename=hierarchy, species="HSA"):
    df = pd.read_csv(filename, sep="\t")
    df.columns = ["source", "target"]
    df = df[df["target"].str.contains(species)]
    return df


def extend(graph, node, n_levels, handle="_copy"):
    """add additional edges to extend n_levels branches """
    edges = []
    source = node
    for level in range(n_levels):
        target = node + handle + str(level + 1)
        edge = (source, target)
        source = target
        edges.append(edge)

    graph.add_edges_from(edges)
    return graph


def extend_direct_connection(graph, n_level=5):
    terminal_nodes = get_nodes_at_level(graph, n_level)

    # get terminal nodes in the gmt file
    df_pathway2gene = load_genes()
    terminal_nodes_all = df_pathway2gene.pathway.unique().tolist()

    # get those needs to be extended and its level in the hierarchy
    terminal_nodes_rewrite = []
    for node in terminal_nodes:
        terminal_nodes_rewrite.append(re.sub('_copy.*', '', node))

    for node in terminal_nodes_all:
        if node not in terminal_nodes_rewrite and node in graph:
            level = nx.shortest_path_length(graph, source="root", target=node)
            num_extension = n_level - level
            if num_extension == 0:
                raise ValueError("Detect terminal nodes here!")
            graph = extend(graph, node, num_extension, handle="_extend")

    return graph


def get_sub_graph(graph, n_levels=5):
    """get subgraph that extend from root with a specified radius"""
    sub_graph = nx.ego_graph(graph, 'root', radius=n_levels)  # n_levels edges away from 'root'
    terminal_nodes = [n for n, d in sub_graph.out_degree() if d == 0]

    # extend those terminal nodes shorter than n_levels
    for node in terminal_nodes:
        distance = len(
            nx.shortest_path(sub_graph, source='root', target=node))  # number of nodes from 'root' to terminal
        if distance <= n_levels:
            diff = n_levels - distance + 1
            sub_graph = extend(sub_graph, node, diff)

    # now all branches are at least n_levels long.

    return sub_graph


def get_nodes_at_level(graph, level):
    """get nodes at n_level"""
    # get all nodes within distance around the query node
    nodes = set(nx.ego_graph(graph, 'root', radius=level))

    # remove nodes that are not **at** the specified distance but closer
    if level >= 1.:
        nodes -= set(nx.ego_graph(graph, 'root', radius=level - 1))

    return list(nodes)


def get_layers_from_graph(graph, n_levels):
    """pathway map for each layer. layers[i] contains a dictionary that map pathway in i-th layer to (i+1)th layer
    1)  layer[1] is the highest biological level pathway.
    2)  layer[n_levels] is the lowest biological level pathway.
    """

    layers, all_node_names = [], []

    for i in range(n_levels):
        node_names = []
        dict = {}
        nodes = get_nodes_at_level(graph, i)
        for node in nodes:
            node_names.append(node)

            next_nodes = graph.successors(node)
            dict[node] = [next_node for next_node in next_nodes]

        layers.append(dict)
        node_names.sort()
        all_node_names.append(node_names)

    last_layer_nodes = [node for node in get_nodes_at_level(graph, n_levels)]
    last_layer_nodes.sort()
    all_node_names.append(last_layer_nodes)

    return layers, all_node_names


def delete_extended_node(graph, identifier, extend_number):
    for i in reversed(range(1, extend_number+1)):
        node_to_be_deleted = identifier + "_extend"+str(i)
        graph.remove_node(node_to_be_deleted)

    return graph


def pathway2gene(graph, df_gene2pathway, lowest_pathways, n_level=5):
    dict= {}

    # for original terminal pathways and pure extension
    genes_connected = []
    for pathway in lowest_pathways:
        if "_extend" not in pathway:
            genes = df_gene2pathway[df_gene2pathway['pathway'] == re.sub('_copy.*', '', pathway)]['gene'].unique()
            dict[pathway] = genes # directly add these genes
            for gene in genes:
                if gene not in genes_connected:
                    genes_connected.append(gene)

    # for higher pathways that have both downstream pathways and direct connections with genes. add gene to "_extend1" pathways first, then goes up!
    for i in range(1, n_level):
        genes_in_this_level = []
        for pathway in lowest_pathways:
            if "_extend"+str(i) in pathway:
                pathway_extended, _ = pathway.split("_")
                genes = df_gene2pathway[df_gene2pathway['pathway'] == pathway_extended]['gene'].unique()
                to_be_added = []
                for gene in genes:
                    if gene not in genes_connected:
                        to_be_added.append(gene)
                    if gene not in genes_in_this_level:
                        genes_in_this_level.append(gene)
                # if no extension is needed in this branch, delete the branch in the graph
                if len(to_be_added) == 0:
                    graph = delete_extended_node(graph, pathway_extended, i)
                else:
                    dict[pathway] = to_be_added


            # don't use those already used in lower pathways!
        genes_connected = [*genes_connected, *genes_in_this_level]

    return graph, dict


def get_linear_mask(higher_pathways, lower_pathways, dict, input_per_lower):
    mask = np.zeros((input_per_lower*len(lower_pathways), len(higher_pathways)))

    for i, lower_pathway in enumerate(lower_pathways):
        for j, higher_pathway in enumerate(higher_pathways):
            if lower_pathway in dict[higher_pathway]:
                for k in range(input_per_lower):
                    mask[(k+1)*i, j] = 1.
    return mask


def get_sparsity(masks):
    sparsity = []
    for mask in masks:
        sparsity.append(1 - np.sum(mask) / mask.size)

    return sparsity


class ReactomeNetwork:
    def __init__(self, ordered_gene_list, unused_branches=None, species="HSA", n_levels=5, fix_connection=True,
                 inputs_per_gene=1):
        print('am I ever called?')
        self.df_hierarchy = load_hierarchy(species=species)
        self.df_pathway_names = load_pathway_encoding(species=species)
        self.df_gene2pathway = load_genes(pathway_col=1, genes_start_col=2)
        self.gene_list = ordered_gene_list
        self.n_levels = n_levels
        self.unused_branches = unused_branches
        self.inputs_per_gene = inputs_per_gene

        print("Calculating graph....")
        self.complete_graph = self.get_reactome_graph()

        self.sub_graph = self.get_sub_graph(n_levels)

        # check for direct connection to higher pathways
        if fix_connection:
            self.sub_graph = extend_direct_connection(self.sub_graph, n_level=n_levels)

        self.layers, self.all_nodes = self.get_layers(n_levels)

        self.num_nodes = self.get_nodes_count(self.all_nodes)

        print("Extracting masks for all layers....")
        self.layer_dfs = self.get_layer_dfs()
        self.masks = self.get_masks(inputs_per_gene)
        self.sparsity = get_sparsity(self.masks)
        self.info()

        print("Done!")

    def get_reactome_graph(self):
        graph = nx.from_pandas_edgelist(self.df_hierarchy, 'source', 'target', create_using=nx.DiGraph())
        graph.name = 'reactome'

        candidates = self.roots(graph)

        roots = candidates

        if self.unused_branches is None:
            # get all branches
            print("Using all {} nodes from the Reactome!".format(len(roots)))
        else:
            # get specific branches
            if type(self.unused_branches) is str:
                assert self.unused_branches in roots
                roots.remove(self.unused_branches)
            elif type(self.unused_branches) is list:
                for branch in self.unused_branches:
                    assert branch in roots
                    roots.remove(branch)
            else:
                raise TypeError("Unsupported data type for branches!")

        return self.add_root_node(graph, roots)

    def get_sub_graph(self, n_levels):
        """get sub-graphs based on the radius (n_levels) counting from 'root' node """
        return get_sub_graph(self.complete_graph, n_levels)

    def get_layers(self, n_levels):
        """get mapping from layer[i] to layer[i+1]"""

        terminal_nodes_temp = get_nodes_at_level(self.sub_graph, n_levels)
        self.sub_graph, gene2pathway_dict = pathway2gene(self.sub_graph, self.df_gene2pathway, terminal_nodes_temp)
        layers, all_nodes = get_layers_from_graph(self.sub_graph, n_levels)

        all_nodes.append(self.gene_list)
        layers.append(gene2pathway_dict)

        return layers, all_nodes
    
    def get_layer_df(self, l, ind):
        df = pd.DataFrame(columns=list(self.layers[l].keys()), index=ind).fillna(0)

        d = self.layers[l]
        for k in df.columns:
            for v in df.index:
                if v in d[k]:
                    df[k][v] = 1

        return df.loc[:, (df!=0).any(0)]
    
    def get_layer_dfs(self):
        layer_dfs = []
        inds = self.gene_list
        for l in range(len(self.layers))[::-1]:
            df = self.get_layer_df(l, inds)
            layer_dfs.append(df)
            inds = df.columns
        return layer_dfs

    def create_mask(self):
        masks = []
        for i in range(self.n_levels + 1):
            if self.inputs_per_gene > 1 and i == self.n_levels:
                mask = get_linear_mask(self.all_nodes[i], self.all_nodes[i+1], self.layers[i], self.inputs_per_gene)
            else:
                mask = get_linear_mask(self.all_nodes[i], self.all_nodes[i+1], self.layers[i], 1)
            print(mask.shape)
            masks.append(mask)

        return masks
    
    def get_masks(self, nbr_genetic_input_types):
        """
        Transforms pd.DataFrame adjacency matrices into binary np.array masks. Input layer connections based on the
         number of genetic inputs.
        :param nbr_genetic_input_types: int; number of genetic input modalities provided
        :return: (list(np.array), list(np.array), np.array); a list of adjacency matrices per layer and a list of
            gene-to-pathway adjacency matrix per layer. The input mask to connect the same gene from different
             modalities to the input node
        """
        pathway_masks = []
        pathway_masks.append(np.concatenate([self.layer_dfs[0].values] * nbr_genetic_input_types))
        for l in self.layer_dfs[1:]:
            pathway_masks.append(l.values)
        return pathway_masks

    def info(self):
        diction = {}
        for i in range(1, self.n_levels + 1):
            diction[i] = [self.num_nodes[i]]

        print("{} |{}".format("Level", "# Nodes"))
        for k, v in diction.items():
            print("  {}   |   {}".format(k, v))

        words = {0: "0-th", 1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
        for i in range(0, self.n_levels + 1):
            print("The {} mask have shape {} with sparsity {}".format(words[i], self.masks[i].shape, self.sparsity[i]))

        print("{}/{} genes are NOT connected to pathways!".
              format(self.calculate_unconnected_genes(self.masks[-1]), len(self.gene_list)))

        print("{}/{} lowest level pathways are NOT connected to genes!".
              format(self.calculate_unconnected_pathway(self.masks[-1]), self.masks[-1].shape[1]))

    @staticmethod
    def roots(graph):
        """get the highest biological pathway hierarchy"""
        return [node for node, in_degree in graph.in_degree() if in_degree == 0]

    @staticmethod
    def add_root_node(graph, root_node):
        edges = [('root', n) for n in root_node]
        graph.add_edges_from(edges)

        return graph

    @staticmethod
    def get_nodes_count(all_nodes):
        return [len(nodes) for nodes in all_nodes]

    @staticmethod
    def calculate_unconnected_genes(mask):

        count = 0
        for i in range(mask.shape[0]):
            if np.sum(mask[i]) < 0.5:
                count += 1

        return count

    @staticmethod
    def calculate_unconnected_pathway(mask):
        """lowest pathway"""
        count = 0
        for i in range(mask.shape[1]):
            if np.sum(mask[:, i]) < 0.5:
                count += 1

        return count