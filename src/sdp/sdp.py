import cvxpy as cp
import numpy as np
import networkx as nx
from src.utils.graph import create_graph
from src.utils.files import list_files
import warnings
import pandas as pd
import time
warnings.filterwarnings("ignore")

df = pd.DataFrame()
run_time = []
cuts = []
graph_name = []
num_edges = []
num_vertices = []

def cut_size(A, T):
    # Define and solve the CVXPY problem.
    # Create a symmetric matrix variable and constraints
    n = A.shape[0]

    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0]
    constraints += [cp.diag(X) == 1]

    prob = cp.Problem(cp.Minimize(cp.trace(A @ X)), constraints)
    prob.solve(solver='CVXOPT')

    # Solve for the maximum cut
    U = np.linalg.cholesky(X.value)

    cut = 0
    for i in range(0, T):
        r = np.random.normal(0, 1, n)
        y = np.sign(U @ r)

        # Calculate the cut
        cut = cut + (np.sum(A) - y.T @ A @ y)/4
    return round(cut / T)


if __name__ == '__main__':

    np.seterr(all="ignore")
    path = "data/max_cut"
    files = list_files(path)
    for file in files:
        graph = create_graph(path, file)
        print(f"[Working with Graph: {file}]\n")
        adj_matrix = nx.adjacency_matrix(graph).toarray()

        try:
            start_cpu_time = time.process_time()
            cut = cut_size(adj_matrix, 100)
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

    df.to_csv("statistics/sdp-1.csv")
