import matplotlib

matplotlib.use('TkAgg')  # Zmiana backendu na TkAgg, aby uniknąć błędu

import networkx as nx
import matplotlib.pyplot as plt
import random
import heapq


def prim_mst(G, start=0):
    """
    Implementacja algorytmu Prima dla grafu G zaczynając od wierzchołka start.
    Zwraca listę krawędzi MST postaci (u, v, waga).
    """
    visited = set([start])
    mst_edges = []
    # Kolejka priorytetowa, elementy: (waga, u, v)
    edges = []

    # Dodaj krawędzie wychodzące z wierzchołka start
    for neighbor, attr in G[start].items():
        heapq.heappush(edges, (attr['weight'], start, neighbor))

    # Dopóki mamy krawędzie i nie odwiedziliśmy wszystkich wierzchołków
    while edges and len(visited) < G.number_of_nodes():
        weight, u, v = heapq.heappop(edges)
        if v in visited:
            continue
        visited.add(v)
        mst_edges.append((u, v, weight))
        # Dodaj nowe krawędzie wychodzące z wierzchołka v
        for neighbor, attr in G[v].items():
            if neighbor not in visited:
                heapq.heappush(edges, (attr['weight'], v, neighbor))

    return mst_edges


def generate_graph(n, p):
    """
    Generuje spójny graf z n wierzchołkami.
    Najpierw tworzy drzewo rozpinające (aby zapewnić spójność), a następnie
    dla każdej pary wierzchołków dodaje krawędź z prawdopodobieństwem p.
    Wagi krawędzi są losowe, w przedziale [1, 20].
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Utwórz drzewo rozpinające: losowa permutacja wierzchołków
    nodes = list(range(n))
    random.shuffle(nodes)
    for i in range(1, n):
        weight = random.randint(1, 20)
        G.add_edge(nodes[i - 1], nodes[i], weight=weight)

    # Dodaj dodatkowe krawędzie z prawdopodobieństwem p
    for i in range(n):
        for j in range(i + 1, n):
            if not G.has_edge(i, j) and random.random() < p:
                weight = random.randint(1, 20)
                G.add_edge(i, j, weight=weight)

    return G


if __name__ == '__main__':
    # Parametry grafu: liczba wierzchołków i prawdopodobieństwo dodania krawędzi
    n = 10
    p = 0.3
    G = generate_graph(n, p)

    # Oblicz MST przy użyciu algorytmu Prima (zaczynamy od wierzchołka 0)
    mst_edges = prim_mst(G, start=0)

    # Utwórz nowy graf zawierający jedynie krawędzie MST
    MST = nx.Graph()
    MST.add_nodes_from(G.nodes())
    for u, v, weight in mst_edges:
        MST.add_edge(u, v, weight=weight)

    # Ustal układ wierzchołków dla wizualizacji
    pos = nx.spring_layout(G, seed=42)

    # Wizualizacja oryginalnego grafu
    plt.figure(figsize=(14, 6))
    plt.subplot(121)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Graf początkowy")

    # Wizualizacja minimalnego drzewa rozpinającego
    plt.subplot(122)
    nx.draw(MST, pos, with_labels=True, node_color='lightgreen', edge_color='red', width=2, node_size=500)
    mst_labels = nx.get_edge_attributes(MST, 'weight')
    nx.draw_networkx_edge_labels(MST, pos, edge_labels=mst_labels)
    plt.title("Minimalne drzewo rozpinające (Algorytm Prima)")

    plt.show()
