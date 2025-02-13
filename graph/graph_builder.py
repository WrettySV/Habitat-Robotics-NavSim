import networkx as nx
import matplotlib.pyplot as plt

class GraphBuilder:
    graph = nx.Graph()

    @classmethod
    def update_graph(cls, visible_objects):
        for obj in visible_objects:
            if obj not in cls.graph:
                cls.graph.add_node(obj)
        for obj1 in visible_objects:
            for obj2 in visible_objects:
                if obj1 != obj2:
                    cls.graph.add_edge(obj1, obj2)

    @classmethod
    def save_graph(cls, file_path="graph/knowledge_graph.png"):
        plt.figure(figsize=(10, 8))
        nx.draw(cls.graph, with_labels=True, node_color='lightblue', edge_color='gray')
        plt.title("Object Knowledge Graph")
        plt.savefig(file_path)
        plt.close()