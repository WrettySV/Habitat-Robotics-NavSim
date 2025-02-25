import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium.spaces import GraphInstance

class GraphBuilder:
    graph = nx.Graph()

    @classmethod
    def update_graph(cls, object_centers): #visible objects but new in order to add new edges


        for obj_id, (y, x) in object_centers.items():
            cls.graph.add_node(obj_id, y=y, x=x)

        object_ids = list(object_centers.keys())
        for i in range(len(object_ids)):
            for j in range(i + 1, len(object_ids)):  
                cls.graph.add_edge(object_ids[i], object_ids[j])

    @classmethod
    def save_graph(cls, file_path="graph/knowledge_graph.png"):
        # Save the graph as an image
        plt.figure(figsize=(10, 8))
        nx.draw(cls.graph, with_labels=True, node_color='lightblue', edge_color='gray')
        plt.title("Object Knowledge Graph")
        plt.savefig(file_path)
        plt.close()


    @classmethod
    def reset_graph(cls):
        # Reset the graph to an empty state
        cls.graph.clear()

    @classmethod
    def to_observation(cls):
        node_attrs = nx.get_node_attributes(cls.graph, "x")
        node_attrs_y = nx.get_node_attributes(cls.graph, "y")
        
        # Get nodes as (num_nodes, 2) array
        nodes = np.array([[node_attrs[n], node_attrs_y[n]] for n in cls.graph.nodes], dtype=np.int32)

        # Get edges (always Discrete(1) in this case)
        edges = np.zeros((len(cls.graph.edges),), dtype=np.int32)  # Single category edges

        # Get edge links as (num_edges, 2) array
        edge_links = np.array(list(cls.graph.edges), dtype=np.int32)

        return GraphInstance(nodes=nodes, edges=edges, edge_links=edge_links)
