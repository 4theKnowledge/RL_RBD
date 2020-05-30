"""
Generates a system DAG for reliability modelling.

@author: Tyler Bikaun
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm   # for node colour map

class SystemGraph:
    def __init__(self, adj_matrix, feat_matrix):
        self.adj_matrix = adj_matrix  # adjacency matrix
        self.feat_matrix = feat_matrix  # feature matrix

        # exe
        self.create()

    def create(self):
        """Creates system graph from adjacency and feature matrices"""

        self.G = nx.from_numpy_matrix(self.adj_matrix, create_using=nx.DiGraph)

    def colour_nodes(self):
        """Function for colouring nodes on graph render"""

        viridis = cm.get_cmap('viridis', self.feat_matrix.shape[1])  # unique colour for each col in F
        self.colour_map = []

        for node in self.G.nodes():
            # Slice F to get node 'type/label':
            node_type = np.argwhere(self.feat_matrix[node, :] == 1)[0][0]
            if node_type == 0:
                self.colour_map.append(viridis.colors[0])
            elif node_type == 1:
                self.colour_map.append(viridis.colors[1])
            elif node_type == 2:
                self.colour_map.append(viridis.colors[2])
            else:
                pass

    def render(self):
        """Renders graph for visualisation"""

        pos = nx.circular_layout(self.G)

        # Generate node colours
        self.colour_nodes()

        # Draw network
        nx.draw_circular(self.G, node_color=self.colour_map, node_size=750, cmap=plt.cm.viridis)
        labels = {i: i + 1 for i in self.G.nodes()}
        nx.draw_networkx_labels(self.G, pos, font_size=15)
        # plt.title(f'Number of nodes oh graph {len(self.G.nodes())}')
        plt.show()

    def add_node(self, node_type):
        """Adds a node to system graph"""

        print(f'Shape before adding node: {self.adj_matrix.shape}')
        # Add n to A
        # col
        self.adj_matrix = np.append(self.adj_matrix, np.ones(shape=(self.adj_matrix.shape[1], 1)), 1)
        print(self.adj_matrix.shape)
        # row
        self.adj_matrix = np.append(self.adj_matrix, np.zeros(shape=(1, self.adj_matrix.shape[1])), 0)

        print(f'Shape after adding node: {self.adj_matrix.shape}')

        # Updating F
        # node types are fixed and indexed from 1
        f_row_node_label = np.zeros((self.feat_matrix.shape[1]))
        f_row_node_label[node_type + 1] = 1
        print(f_row_node_label)
        self.feat_matrix = np.append(self.feat_matrix, [f_row_node_label], 0)

    def print_graph_details(self):
        """Outputs graph information"""

        print(f'Nodes')
        print(f'index\tlabel')
        for node_idx in self.G.nodes():
            label = np.argmax(self.feat_matrix[node_idx, :])
            print(f'{node_idx}\t\t{label}')

        print(f'Edges')
        for edge in self.G.edges():
            print(edge)

    def get_node_details(self):
        """Returns a dictionary of nodes and their types"""

        node_dict = {}
        for node_idx in self.G.nodes():
            label = np.argmax(self.feat_matrix[node_idx, :])
            node_dict[node_idx] = label
        return node_dict


def main():
    """

    """
    # A - adjacency matrix (node and edges)
    # an entire row of 1s indicates arrows away FROM the node,
    # an entire column of 1s indicates arrows TO the node
    A = np.array([[0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0]])  # series configuration

    # F - feature list (node type)
    # row indicates node
    # col indicates node type
    F = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0],
                  [1, 0, 0],
                  [1, 0, 0],
                  [1, 0, 0]])

    sys_graph = SystemGraph(A, F)
    sys_graph.print_graph_details()
    node_dict = sys_graph.get_node_details()
    # print(node_dict)
    # sys_graph.render()

    """
    sys_graph.add_node(0)
    sys_graph.create()
    sys_graph.render()
    """


if __name__ == '__main__':
    main()