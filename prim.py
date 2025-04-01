import matplotlib
import networkx as nx
import matplotlib.pyplot as plt
import random
import heapq

matplotlib.use('TkAgg')  # Zmiana backendu na TkAgg, aby uniknąć błędu


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


def dijkstra(G, start=0):
    """
    Implementacja algorytmu Dijkstry na grafie G.
    Zwraca dwa słowniki:
      - distances: najkrótsza odległość od wierzchołka start do każdego wierzchołka,
      - previous: poprzednik każdego wierzchołka na najkrótszej ścieżce.
    """
    distances = {node: float('inf') for node in G.nodes()}
    previous = {node: None for node in G.nodes()}
    distances[start] = 0
    queue = [(0, start)]

    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if current_distance > distances[current_node]:
            continue
        for neighbor, attr in G[current_node].items():
            weight = attr['weight']
            new_distance = current_distance + weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous[neighbor] = current_node
                heapq.heappush(queue, (new_distance, neighbor))
    return distances, previous


def reconstruct_path(previous, start, target):
    """
    Odtwarza najkrótszą ścieżkę od wierzchołka start do target przy użyciu słownika previous.
    """
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    return path


def kruskal_mst(G):
    """
    Implementacja algorytmu Kruskala dla grafu G.
    Zwraca listę krawędzi MST w postaci (u, v, waga).
    """
    # Inicjalizacja struktury union-find
    parent = {node: node for node in G.nodes()}
    rank = {node: 0 for node in G.nodes()}

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(u, v):
        # Połącz dwa zbiory, zwracając True, jeśli połączenie nastąpiło
        root_u = find(u)
        root_v = find(v)
        if root_u == root_v:
            return False  # u i v są już w tym samym zbiorze
        if rank[root_u] < rank[root_v]:
            parent[root_u] = root_v
        elif rank[root_u] > rank[root_v]:
            parent[root_v] = root_u
        else:
            parent[root_v] = root_u
            rank[root_u] += 1
        return True

    # Sortujemy krawędzie według wagi
    sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'])
    mst_edges = []

    # Wybieramy krawędzie, które nie tworzą cyklu
    for u, v, data in sorted_edges:
        if union(u, v):
            mst_edges.append((u, v, data['weight']))
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


def convert_graph_to_adj_dict(G):
    """
    Przekształca graf G (reprezentowany przez networkx) do postaci słownika sąsiedztwa.
    """
    adj_dict = {}
    for node in G.nodes():
        # Tworzymy słownik sąsiadów dla bieżącego wierzchołka
        neighbors = {}
        for neighbor, data in G[node].items():
            # Pobieramy wagę krawędzi, zakładając, że klucz 'weight' jest ustawiony
            neighbors[neighbor] = data.get('weight')
        adj_dict[node] = neighbors
    return adj_dict






def prim_mst_from_adj(adj, start=None):
    """
    Wykonuje algorytm Prima na grafie reprezentowanym przez słownik sąsiedztwa.
    """
    if start is None:
        start = next(iter(adj))  # Wybieramy pierwszy klucz jako wierzchołek startowy

    visited = {start}
    mst_edges = []  # Lista krawędzi MST w postaci (u, v, weight)
    pq = []         # Kolejka priorytetowa: (waga, wierzchołek źródłowy, wierzchołek docelowy)

    # Dodajemy krawędzie wychodzące z wierzchołka startowego
    for neighbor, weight in adj[start].items():
        heapq.heappush(pq, (weight, start, neighbor))

    while pq and len(visited) < len(adj):
        weight, u, v = heapq.heappop(pq)
        if v in visited:
            continue
        visited.add(v)
        mst_edges.append((u, v, weight))
        # Dodajemy nowe krawędzie wychodzące z wierzchołka v
        for neighbor, w in adj[v].items():
            if neighbor not in visited:
                heapq.heappush(pq, (w, v, neighbor))

    # Tworzymy słownik sąsiedztwa dla MST
    mst_adj = {node: {} for node in adj}
    for u, v, weight in mst_edges:
        mst_adj[u][v] = weight
        mst_adj[v][u] = weight  # Ponieważ graf jest nieskierowany
    return mst_adj




def calculate_mst_cost_dict(mst_adj_dict):
    """
    Oblicza całkowity koszt minimalnego drzewa rozpinającego
    reprezentowanego jako słownik sąsiedztwa.
    """
    total_cost = 0
    visited_edges = set()  # Zbiór do zapamiętania już uwzględnionych krawędzi

    for u, neighbors in mst_adj_prim.items():
        for v, weight in neighbors.items():
            edge = tuple(sorted((u, v)))  # Standaryzacja reprezentacji krawędzi
            if edge not in visited_edges:
                visited_edges.add(edge)
                total_cost += weight
    return total_cost


def kruskal_mst_from_adj(graph):
    # Generowanie listy unikalnych krawędzi
    edges = []
    for u in graph:
        for v, weight in graph[u].items():
            if u < v:  # Zapobiega duplikatom
                edges.append((weight, u, v))

    # Sortowanie krawędzi rosnąco według wagi
    edges.sort()

    # Inicjalizacja struktur dla Union-Find
    vertices = list(graph.keys())
    parent = {v: v for v in vertices}
    rank = {v: 0 for v in vertices}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Kompresja ścieżki
            u = parent[u]
        return u

    mst = []
    for edge in edges:
        weight, u, v = edge
        root_u = find(u)
        root_v = find(v)

        if root_u != root_v:
            mst.append((u, v, weight))
            # Union z uwzględnieniem rangi
            if rank[root_u] > rank[root_v]:
                parent[root_v] = root_u
            else:
                parent[root_u] = root_v
                if rank[root_u] == rank[root_v]:
                    rank[root_v] += 1
            # Przerwij jeśli mamy już n-1 krawędzi
            if len(mst) == len(vertices) - 1:
                break

    mst_graph = {}
    for u, v, weight in mst:
        mst_graph.setdefault(u, {})[v] = weight
        mst_graph.setdefault(v, {})[u] = weight

    sorted_nodes = sorted(mst_graph.keys(), key=lambda x: int(x))
    sorted_mst = {node: mst_graph[node] for node in sorted_nodes}

    return sorted_mst


def dijkstra_mst_from_adj(graph, start):
    costs = {node: float('inf') for node in graph}
    predecessors = {node: None for node in graph}
    costs[start] = 0

    heap = [(0, start)]

    while heap:
        current_cost, current_node = heapq.heappop(heap)

        if current_cost > costs[current_node]:
            continue

        for neighbor, edge_data in graph[current_node].items():
            weight = edge_data if isinstance(edge_data, (int, float)) else edge_data.get('weight', 0)
            new_cost = current_cost + weight

            if new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                predecessors[neighbor] = current_node
                heapq.heappush(heap, (new_cost, neighbor))

    return costs, predecessors


def convert_networkx_to_edge_list(G):
    """
    Konwertuje graf zapisany w networkx do formatu listy krawędzi:
      [wierzchołek, waga, wierzchołek].
    """
    edge_list = []
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', None)
        edge_list.append([u, weight, v])
    return edge_list




def prim_mst_edge_list(edge_list, start=None):
    """
    Wykonuje algorytm Prima na grafie przedstawionym jako tablica dwuwymiarowa:
    """
    # Ustalamy zbiór wierzchołków występujących w grafie
    vertices = set()
    for edge in edge_list:
        u, weight, v = edge
        vertices.add(u)
        vertices.add(v)

    if start is None:
        start = next(iter(vertices))

    visited = {start}
    mst_edges = []

    # Dopóki nie odwiedzimy wszystkich wierzchołków
    while len(visited) < len(vertices):
        candidate_edge = None
        candidate_weight = float('inf')

        # Przeszukujemy wszystkie krawędzie, aby znaleźć tę o najmniejszej wadze,
        # która łączy odwiedzony wierzchołek z nieodwiedzonym.
        for edge in edge_list:
            u, weight, v = edge
            if ((u in visited and v not in visited) or (
                    v in visited and u not in visited)) and weight < candidate_weight:
                candidate_edge = edge
                candidate_weight = weight

        # Jeśli nie znaleziono krawędzi, graf jest niespójny
        if candidate_edge is None:
            raise ValueError("Graf jest niespójny!")

        # Dodajemy krawędź do MST
        mst_edges.append(candidate_edge)

        # Dodajemy do odwiedzonych nowy wierzchołek (ten, który nie był jeszcze odwiedzony)
        u, weight, v = candidate_edge
        if u in visited and v not in visited:
            visited.add(v)
        elif v in visited and u not in visited:
            visited.add(u)

    return mst_edges




def cost_edge_list(edges):
    return sum(waga for _, waga, _ in edges)


def kruskal_mst_edge_list(edge_list):
    """
    Wykonuje algorytm Kruskala na grafie przedstawionym jako lista krawędzi w formacie:
      [wierzchołek, waga, wierzchołek].
    """
    # Wyznaczamy zbiór wszystkich wierzchołków w grafie
    vertices = set()
    for edge in edge_list:
        u, weight, v = edge
        vertices.add(u)
        vertices.add(v)

    # Inicjalizacja struktury union-find
    parent = {v: v for v in vertices}
    rank = {v: 0 for v in vertices}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX == rootY:
            return False
        if rank[rootX] < rank[rootY]:
            parent[rootX] = rootY
        elif rank[rootX] > rank[rootY]:
            parent[rootY] = rootX
        else:
            parent[rootY] = rootX
            rank[rootX] += 1
        return True

    # Sortujemy krawędzie według wagi (rosnąco)
    sorted_edges = sorted(edge_list, key=lambda e: e[1])

    mst_edges = []
    for edge in sorted_edges:
        u, weight, v = edge
        # Jeśli połączenie u i v nie tworzy cyklu, dodajemy krawędź do MST
        if union(u, v):
            mst_edges.append(edge)

    return mst_edges




def dijkstra_edge_list(edge_list, start=0):
    """
    Wykonuje algorytm Dijkstry na grafie reprezentowanym bezpośrednio jako tablica tablic
    """
    # Wyznaczamy zbiór wszystkich wierzchołków
    vertices = set()
    for u, cost, v in edge_list:
        vertices.add(u)
        vertices.add(v)

    # Inicjalizacja odległości i poprzedników
    distances = {v: float('inf') for v in vertices}
    previous = {v: None for v in vertices}
    distances[start] = 0

    # Kolejka priorytetowa: (koszt dotarcia, wierzchołek)
    pq = [(0, start)]
    visited = set()

    while pq:
        current_cost, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        # Przeglądamy wszystkie krawędzie w liście, aby znaleźć te wychodzące z u
        for a, cost, b in edge_list:
            # Sprawdzamy, czy krawędź wychodzi z u (w obu kierunkach, bo graf jest nieskierowany)
            if a == u and b not in visited:
                new_cost = current_cost + cost
                if new_cost < distances[b]:
                    distances[b] = new_cost
                    previous[b] = u
                    heapq.heappush(pq, (new_cost, b))
            elif b == u and a not in visited:
                new_cost = current_cost + cost
                if new_cost < distances[a]:
                    distances[a] = new_cost
                    previous[a] = u
                    heapq.heappush(pq, (new_cost, a))

    return distances, previous


def reconstruct_shortest_paths_edge_list(distances, previous):
    """
    Rekonstruuje najkrótsze ścieżki dla wszystkich wierzchołków na podstawie
    słowników distances i previous.
    """
    result = []
    for vertex in sorted(distances.keys()):
        # Rekonstruujemy ścieżkę od startu do wierzchołka `vertex`
        path = []
        current = vertex
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()  # Odwracamy, aby ścieżka była od startu do celu

        result.append(f"Do wierzchołka {vertex}: koszt = {distances[vertex]}, ścieżka = {path}")
    return result




if __name__ == '__main__':
    # Parametry grafu: liczba wierzchołków i prawdopodobieństwo dodania krawędzi
    n = 10
    p = 0.3
    G = generate_graph(n, p)

    # Oblicz MST przy użyciu algorytmu Prima (zaczynamy od wierzchołka 0)
    mst_edges = prim_mst(G, start=0)

    # Obliczenie całkowitego kosztu MST (suma wag krawędzi)
    total_cost = sum(weight for _, _, weight in mst_edges)
    print("Całkowity koszt MST (Prim NetworkX):", total_cost)

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

    # Uruchamiamy algorytm Kruskala na tym samym grafie G
    mst_edges_kruskal = kruskal_mst(G)

    # Obliczenie całkowitego kosztu MST (suma wag krawędzi)
    total_cost_kruskal = sum(weight for _, _, weight in mst_edges_kruskal)
    print("Całkowity koszt MST (Kruskal NetworkX):", total_cost_kruskal)

    # Budujemy graf reprezentujący MST uzyskane algorytmem Kruskala
    MST_kruskal = nx.Graph()
    MST_kruskal.add_nodes_from(G.nodes())
    for u, v, weight in mst_edges_kruskal:
        MST_kruskal.add_edge(u, v, weight=weight)

    # Wizualizacja: oryginalny graf oraz MST (Kruskal)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(14, 6))

    # Lewy panel: Oryginalny graf
    plt.subplot(121)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Graf początkowy")

    # Prawy panel: MST uzyskane algorytmem Kruskala
    plt.subplot(122)
    nx.draw(MST_kruskal, pos, with_labels=True, node_color='lightgreen', edge_color='red', node_size=500, width=2)
    mst_edge_labels = nx.get_edge_attributes(MST_kruskal, 'weight')
    nx.draw_networkx_edge_labels(MST_kruskal, pos, edge_labels=mst_edge_labels)
    plt.title("Minimalne drzewo rozpinające (Kruskal)")

    plt.show()



    # Wykonanie algorytmu Dijkstry na tym samym grafie G
    start_node = 0
    distances, previous = dijkstra(G, start_node)

    # Odtwarzamy ścieżki dla każdego wierzchołka
    shortest_paths = {}
    for node in G.nodes():
        shortest_paths[node] = reconstruct_path(previous, start_node, node)

    # Wypisanie wyników w konsoli
    print("Najkrótsze ścieżki z wierzchołka (NetworkX) ", start_node)
    for node in G.nodes():
        print(f"Do wierzchołka {node}: koszt = {distances[node]}, ścieżka = {shortest_paths[node]}")

    # Wizualizacja grafu z nałożonym drzewem najkrótszych ścieżek (SPT)
    SPT = nx.Graph()
    SPT.add_nodes_from(G.nodes())
    for node in G.nodes():
        if previous[node] is not None:
            SPT.add_edge(previous[node], node, weight=G[previous[node]][node]['weight'])

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(14, 6))

    # Lewy panel: graf z nałożonymi najkrótszymi ścieżkami (SPT na czerwono)
    plt.subplot(121)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
    nx.draw_networkx_edges(SPT, pos, edge_color='red', width=2)
    plt.title("Graf z najkrótszymi ścieżkami (Dijkstra)")

    # Prawy panel: tabela wyników
    plt.subplot(122)
    plt.axis('tight')
    plt.axis('off')
    table_data = [[node, distances[node], shortest_paths[node]] for node in sorted(G.nodes())]
    table_columns = ["Wierzchołek", "Koszt", "Ścieżka"]
    plt.table(cellText=table_data, colLabels=table_columns, loc='center')
    plt.title("Tabela najkrótszych ścieżek")

    plt.show()

    adj_dict = convert_graph_to_adj_dict(G)
    mst_adj_prim = prim_mst_from_adj(adj_dict)
    print("\nPrim słownik sąsiedztwa minimalnego drzewa: ", mst_adj_prim)
    cost_mst_adj = calculate_mst_cost_dict(mst_adj_prim)
    print("Całkowity koszt MST (Prim Słownik sąsiedztwa): ", cost_mst_adj)

    mst_adj_kruskal = kruskal_mst_from_adj(adj_dict)
    print("Kruskal słownik sąsiedztwa minimalnego drzewa: ", mst_adj_kruskal)
    cost_mst_adj_kruskal = calculate_mst_cost_dict(mst_adj_kruskal)
    print("Całkowity koszt MST (Kruskal Słownik sąsiedztwa): ", cost_mst_adj_kruskal)

    start_node = 0
    distances, previous = dijkstra_mst_from_adj(adj_dict, start_node)

    # Odtwarzamy ścieżki dla każdego wierzchołka
    shortest_paths = {}
    for node in G.nodes():
        shortest_paths[node] = reconstruct_path(previous, start_node, node)

    # Wypisanie wyników w konsoli
    print("Najkrótsze ścieżki z wierzchołka (Słownik sąsiedztwa) ", start_node)
    for node in G.nodes():
        print(f"Do wierzchołka {node}: koszt = {distances[node]}, ścieżka = {shortest_paths[node]}")

    # trzeci sposob zapisu grafów
    # [wierzchołek, waga, wierzchołek])
    G_edge_list = convert_networkx_to_edge_list(G)
    edge_list_mst_prim = prim_mst_edge_list(G_edge_list)
    cost_edge_list_mst_prim = cost_edge_list(edge_list_mst_prim)
    print("\nCałkowity koszt MST (Prim Tablica krawędzi): ", cost_edge_list_mst_prim)
    edge_list_mst_kruskal = kruskal_mst_edge_list(G_edge_list)
    cost_edge_list_mst_kruskal = cost_edge_list(edge_list_mst_kruskal)
    print("Całkowity koszt MST (Kruskal Tablica krawędzi): ", cost_edge_list_mst_kruskal)
    shortest_paths_edge_list_distances, shortest_paths_edge_list_previous = dijkstra_edge_list(G_edge_list, start_node)
    paths = reconstruct_shortest_paths_edge_list(shortest_paths_edge_list_distances, shortest_paths_edge_list_previous)
    print("Najkrótsze ścieżki z wierzchołka (Tablica krawędzi) ", start_node)
    for line in paths:
        print(line)
