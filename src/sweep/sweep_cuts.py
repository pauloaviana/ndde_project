import numpy as np
import pandas as pd
import networkx as nx
import warnings
import time
from src.utils.graph import create_graph
from src.utils.files import list_files
from src.trevisan.functions import *

df = pd.DataFrame()
run_time = []
cuts = []
graph_name = []
num_edges = []
num_vertices = []

def sweep(adj_matrix, n, adj_list):
    D = adj_matrix_deg_matrix(adj_matrix, n)
    neg_sqrt_D = np.zeros([n, n])
    for i in range(n):
        if D[i][i] == 0:
            neg_sqrt_D[i][i] = 0
        else:
            neg_sqrt_D[i, i] = D[i, i] ** (-1 / 2)
    norm_mat = neg_sqrt_D @ adj_matrix @ neg_sqrt_D
    x = smallest_eigenvector(norm_mat + np.identity(n))

    p = np.argsort(x)
    max_val = 0
    max_cut = []

    for i in range(n):
        val = cut_value(adj_matrix, p[:i], adj_list)
        if val > max_val:
            max_val = val
            partition = p[:i]
    return max_val

if __name__ == '__main__':

    np.seterr(all="ignore")
    path = "data/erdos_renyi"
    files = list_files(path)
    for file in files:
        graph = create_graph(path, file)
        print(f"[Working with Graph: {file}]\n")
        adj_matrix = nx.adjacency_matrix(graph).toarray()

        try:
            start_cpu_time = time.process_time()
            cut = sweep(adj_matrix, len(adj_matrix), get_adj_list(adj_matrix))
            end_cpu_time = time.process_time()

            clock = end_cpu_time - start_cpu_time
            run_time.append(clock)

            print(f"\nTime = {clock}")
           
            cuts.append(cut)
            print(f"Number of Cuts = {cut}")

            graph_name.append(file)
            print(f"graph = {file}")

            num_vertices.append(graph.number_of_nodes())
            print(f"Number of Vertices = {graph.number_of_nodes()}")

            num_edges.append(graph.number_of_edges())
            print(f"Number of Edges = {graph.number_of_edges()}\n")
            

        except Exception as e:
            print(e)

    df['graph'] = graph_name
    df['cuts'] = cuts
    df['vertices'] = num_vertices
    df['edges'] = num_edges
    df['time'] = run_time

    df.to_csv("statistics/sweep_cuts.csv")
