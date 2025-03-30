from prim import *
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')  # Zmiana backendu na TkAgg, aby uniknąć błędu

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

        #Networkx
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

        #Tablica krawędzi
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



    print("Prim (networkx): " + str(primTime1))
    print("Prim (słownik): " + str(primTime2))
    print("Prim (tablica krawędzi): " + str(primTime3))
    print("Kruskal (networkx): " + str(kruskalTime1))
    print("Kruskal (słownik): " + str(kruskalTime2))
    print("Kruskal (tablica krawędzi): " + str(kruskalTime3))
    print("Dijkstra (networkx): " + str(dijkstraTime1))
    print("Dijkstra (słownik): " + str(dijkstraTime2))
    print("Dijkstra (tablica krawędzi): " + str(dijkstraTime3))

    names = ["Prim", "Kruskal", "Dijkstra"]
    # names = ["Networkx", "Słownik", "Tablica krawędzi"]
    networkxTimes = [primTime1, kruskalTime1, dijkstraTime1]
    slownikTimes = [primTime2, kruskalTime2, dijkstraTime2]
    edgeTimes = [primTime3, kruskalTime3, dijkstraTime3]
    # primVals = [primTime1, primTime2, primTime3]
    # kruskalVals = [kruskalTime1, kruskalTime2, kruskalTime3]
    # dijkstraVals = [dijkstraTime1, dijkstraTime2, dijkstraTime3]
    # vals = [primTime1, primTime2, primTime3, kruskalTime1, kruskalTime2, kruskalTime3, dijkstraTime1, dijkstraTime2, dijkstraTime3]

    # plt.bar(names, vals)
    # plt.title('Results')
    # plt.xlabel('Method')
    # plt.ylabel('Time')
    # plt.show()

    # plt.bar(names, primVals)
    # plt.title('Prim Results')
    # plt.xlabel('Method')
    # plt.ylabel('Time')
    # plt.show()
    #
    # plt.bar(names, kruskalVals)
    # plt.title('Kruskal Results')
    # plt.xlabel('Method')
    # plt.ylabel('Time')
    # plt.show()
    #
    # plt.bar(names, dijkstraVals)
    # plt.title('Dijkstra Results')
    # plt.xlabel('Method')
    # plt.ylabel('Time')
    # plt.show()

    X_axis = np.arange(len(names))

    plt.bar(X_axis - 0.2, networkxTimes, 0.2, label='Networkx')
    plt.bar(X_axis, slownikTimes, 0.2, label='Słownik')
    plt.bar(X_axis + 0.2, edgeTimes, 0.2, label='Tablica krawędzi')

    plt.xticks(X_axis, names)
    plt.xlabel("Algorytm")
    plt.ylabel("Czas")
    plt.legend()
    plt.show()
