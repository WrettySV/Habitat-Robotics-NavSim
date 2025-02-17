import networkx as nx
import matplotlib.pyplot as plt

class GraphBuilder:
    # Class-level graph instance (shared by all instances of GraphBuilder)
    graph = nx.Graph()

    @classmethod
    def update_graph(cls, visible_objects):
        visible_objects = list(visible_objects)  

        cls.graph.add_nodes_from(visible_objects)

        for i in range(len(visible_objects)):
            for j in range(i + 1, len(visible_objects)):  # Avoid duplicates
                cls.graph.add_edge(visible_objects[i], visible_objects[j])

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
