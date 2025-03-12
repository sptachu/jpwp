import matplotlib
import networkx as nx
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# Tworzymy pusty graf nieskierowany
G = nx.Graph()

# Dodajemy węzły
G.add_node("A")
G.add_nodes_from(["B", "C", "D"])

# Dodajemy krawędzie
G.add_edge("A", "B")
G.add_edges_from([("A", "C"), ("B", "D")])

# Rysowanie grafu
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', font_weight='bold')
plt.show()
