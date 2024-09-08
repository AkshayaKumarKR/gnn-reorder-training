import networkx as nx
import matplotlib.pyplot as plt

# Create a new graph
G = nx.Graph()

# Add nodes
nodes = [1, 2, 3, 4, 5]
G.add_nodes_from(nodes)

# Add edges
edges = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 5), (3, 4), (4, 5)]
G.add_edges_from(edges)

# Draw the graph
pos = nx.spring_layout(G)  # Position nodes using the Fruchterman-Reingold force-directed algorithm
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=16, font_weight='bold')
plt.title("Simple Graph with 5 Nodes and Multiple Edges")
plt.show()