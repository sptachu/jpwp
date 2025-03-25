import time

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
        # Znajdź reprezentanta zbioru z kompresją ścieżki
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

    Zwraca:
      adj_dict - słownik, w którym kluczami są wierzchołki, a wartościami słowniki reprezentujące
                 sąsiadów danego wierzchołka wraz z wagami krawędzi.

    Przykład wyniku:
      {
          0: {1: 5, 2: 3},
          1: {0: 5, 3: 2},
          2: {0: 3},
          3: {1: 2}
      }
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


# Przykładowe użycie:
# adj_dict = convert_graph_to_adj_dict(G)
# print(adj_dict)



def prim_mst_from_adj(adj, start=None):
    """
    Wykonuje algorytm Prima na grafie reprezentowanym przez słownik sąsiedztwa.

    Argumenty:
      - adj: słownik sąsiedztwa reprezentujący graf, np.
             {0: {1: 5, 2: 3}, 1: {0: 5, 3: 2}, ...}
      - start: opcjonalny wierzchołek startowy; jeśli nie podany, wybierany jest pierwszy klucz ze słownika.

    Zwraca:
      - mst_adj: słownik sąsiedztwa reprezentujący minimalne drzewo rozpinające (MST).
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

# Przykładowe użycie:
# mst_adj = prim_mst_from_adj(adj_dict)   # gdzie adj_dict to słownik sąsiedztwa wygenerowanego grafu
# print(mst_adj)


def calculate_mst_cost_dict(mst_adj_dict):
    """
    Oblicza całkowity koszt minimalnego drzewa rozpinającego
    reprezentowanego jako słownik sąsiedztwa.

    Argument:
      - mst_adj: słownik sąsiedztwa reprezentujący MST, gdzie każda krawędź
                 pojawia się w obu kierunkach.

    Zwraca:
      - total_cost: suma wag wszystkich krawędzi MST (każda krawędź liczona raz).
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

    # Funkcja find z kompresją ścieżki
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

    # Konwersja na słownik sąsiedztwa
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

        # Zmiana w sposobie pobierania wag - kluczowa poprawka
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

    Zakłada, że graf jest nieskierowany. Dla krawędzi, które nie posiadają atrybutu 'weight',
    przypisywana jest wartość None.

    Argument:
      - G: graf utworzony przy użyciu biblioteki networkx.

    Zwraca:
      - edge_list: lista krawędzi w formacie [u, weight, v].
    """
    edge_list = []
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', None)
        edge_list.append([u, weight, v])
    return edge_list


# Przykładowe użycie:
# edge_list = convert_networkx_to_edge_list(G)
# print(edge_list)


def prim_mst_edge_list(edge_list, start=None):
    """
    Wykonuje algorytm Prima na grafie przedstawionym jako tablica dwuwymiarowa:
      [[u, waga, v], [u, waga, v], ...]

    Funkcja pracuje bezpośrednio na podanej tablicy krawędzi, nie konwertując jej do innej reprezentacji.

    Zwraca:
      mst_edges - lista krawędzi MST w tym samym formacie.
    """
    # Ustalamy zbiór wierzchołków występujących w grafie
    vertices = set()
    for edge in edge_list:
        u, weight, v = edge
        vertices.add(u)
        vertices.add(v)

    # Jeśli nie podano wierzchołka startowego, wybieramy dowolny z grafu
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


# Przykładowe użycie:
# edge_list = [
#     [0, 5, 1],
#     [0, 3, 2],
#     [1, 2, 2],
#     [1, 6, 3],
#     [2, 7, 3],
#     [2, 4, 4],
#     [3, 1, 4]
# ]
# mst = prim_mst_direct(edge_list)
# print("Krawędzie MST:", mst)


def cost_edge_list(edges):
    return sum(waga for _, waga, _ in edges)


def kruskal_mst_edge_list(edge_list):
    """
    Wykonuje algorytm Kruskala na grafie przedstawionym jako lista krawędzi w formacie:
      [wierzchołek, waga, wierzchołek].

    Zwraca:
      mst_edges - lista krawędzi należących do MST, w tym samym formacie.
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


# Przykładowe użycie:
# edge_list = [
#     [0, 5, 1],
#     [0, 3, 2],
#     [1, 2, 2],
#     [1, 6, 3],
#     [2, 7, 3],
#     [2, 4, 4],
#     [3, 1, 4]
# ]
# mst = kruskal_mst_from_edge_list(edge_list)
# print("Krawędzie MST (Kruskal):", mst)


def dijkstra_edge_list(edge_list, start=0):
    """
    Wykonuje algorytm Dijkstry na grafie reprezentowanym bezpośrednio jako lista krawędzi:
      [wierzchołek, koszt, wierzchołek]

    Argumenty:
      - edge_list: lista krawędzi, np. [[0, 5, 1], [0, 3, 2], [1, 2, 2], ...]
      - start: wierzchołek startowy

    Zwraca:
      - distances: słownik, gdzie kluczem jest wierzchołek, a wartością najkrótszy koszt dotarcia od 'start'
      - previous: słownik poprzedników na najkrótszej ścieżce (umożliwiający odtworzenie ścieżki)
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


# Przykładowe użycie:
# edge_list = [
#     [0, 5, 1],
#     [0, 3, 2],
#     [1, 2, 2],
#     [1, 6, 3],
#     [2, 7, 3],
#     [2, 4, 4],
#     [3, 1, 4]
# ]
# start_vertex = 0
# distances, previous = dijkstra_direct(edge_list, start_vertex)
# print("Najkrótsze odległości:", distances)
# print("Poprzednicy:", previous)


def reconstruct_shortest_paths_edge_list(distances, previous):
    """
    Rekonstruuje najkrótsze ścieżki dla wszystkich wierzchołków na podstawie
    słowników distances i previous.

    Zwraca listę napisów w formacie:
      "Do wierzchołka X: koszt = Y, ścieżka = [lista wierzchołków]"
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


# Przykładowe użycie:
# distances, previous = dijkstra_direct(edge_list, start_vertex)
# paths = reconstruct_shortest_paths(distances, previous)
# for line in paths:
#     print(line)


if __name__ == '__main__':
    primTime1 = 0
    kruskalTime1 = 0
    dijkstraTime1 = 0
    primTime2 = 0
    kruskalTime2 = 0
    dijkstraTime2 = 0
    primTime3 = 0
    kruskalTime3 = 0
    dijkstraTime3 = 0
    for i in range (0,1000):
        # Parametry grafu: liczba wierzchołków i prawdopodobieństwo dodania krawędzi
        n = 100
        p = 0.3
        G = generate_graph(n, p)

        #Kopiec
        startTimePrim1 = time.time()
        mst_edges = prim_mst(G, start=0)
        primTime1 = primTime1 + time.time() - startTimePrim1

        startTimeKruskal1 = time.time()
        mst_edges_kruskal = kruskal_mst(G)
        kruskalTime1 = kruskalTime1 + time.time() - startTimeKruskal1

        startTimeDijkstra1 = time.time()
        distances, previous = dijkstra(G, 0)
        dijkstraTime1 = dijkstraTime1 + time.time() - startTimeDijkstra1

        #Słownik sąsiedztwa
        adj_dict = convert_graph_to_adj_dict(G)

        startTimePrim2 = time.time()
        mst_adj_prim = prim_mst_from_adj(adj_dict)
        primTime2 = primTime2 + time.time() - startTimePrim2

        startTimeKruskal2 = time.time()
        mst_adj_kruskal = kruskal_mst_from_adj(adj_dict)
        kruskalTime2 = kruskalTime2 + time.time() - startTimeKruskal2

        startTimeDijkstra2 = time.time()
        distances2, previous2 = dijkstra_mst_from_adj(adj_dict, 0)
        dijkstraTime2 = dijkstraTime2 + time.time() - startTimeDijkstra2

        #Trzeci sposób
        G_edge_list = convert_networkx_to_edge_list(G)

        startTimePrim3 = time.time()
        edge_list_mst_prim = prim_mst_edge_list(G_edge_list)
        primTime3 = primTime3 + time.time() - startTimePrim3

        startTimeKruskal3 = time.time()
        edge_list_mst_kruskal = kruskal_mst_edge_list(G_edge_list)
        kruskalTime3 = kruskalTime3 + time.time() - startTimeKruskal3

        startTimeDijkstra3 = time.time()
        shortest_paths_edge_list_distances, shortest_paths_edge_list_previous = dijkstra_edge_list(G_edge_list)
        dijkstraTime3 = dijkstraTime3 + time.time() - startTimeDijkstra3



    print("Prim (kopiec): " + str(primTime1))
    print("Prim (słownik): " + str(primTime2))
    print("Prim (trzeci sposób): " + str(primTime3))
    print("Kruskal (kopiec): " + str(kruskalTime1))
    print("Kruskal (słownik): " + str(kruskalTime2))
    print("Kruskal (trzeci sposób): " + str(kruskalTime3))
    print("Dijkstra (kopiec): " + str(dijkstraTime1))
    print("Dijkstra (słownik): " + str(dijkstraTime2))
    print("Dijkstra (trzeci sposób): " + str(dijkstraTime3))

    names = ["Kopiec", "Słownik", "Trzeci sposób"]
    primVals = [primTime1, primTime2, primTime3]
    kruskalVals = [kruskalTime1, kruskalTime2, kruskalTime3]
    dijkstraVals = [dijkstraTime1, dijkstraTime2, dijkstraTime3]

    plt.bar(names, primVals)
    plt.title('Prim Results')
    plt.xlabel('Method')
    plt.ylabel('Time')
    plt.show()

    plt.bar(names, kruskalVals)
    plt.title('Kruskal Results')
    plt.xlabel('Method')
    plt.ylabel('Time')
    plt.show()

    plt.bar(names, dijkstraVals)
    plt.title('Dijkstra Results')
    plt.xlabel('Method')
    plt.ylabel('Time')
    plt.show()
