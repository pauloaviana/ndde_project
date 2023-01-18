import cvxpy as cp
import numpy as np
import networkx as nx
from src.utils.graph import create_graph
from src.utils.files import list_files
import warnings
warnings.filterwarnings("ignore")


def cut_size(A, T):
    # Define and solve the CVXPY problem.
    # Create a symmetric matrix variable and constraints
    n = A.shape[0]

    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0]
    constraints += [cp.diag(X) == 1]

    prob = cp.Problem(cp.Minimize(cp.trace(A @ X)), constraints)
    prob.solve(verbose=False)

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
    path = "../../data/erdos_renyi"
    files = list_files(path)
    for file in files:
        graph = create_graph(path, file)
        print(f"[Working with Graph: {file}]\n")
        adj_matrix = nx.adjacency_matrix(graph).toarray()

        try:
            cut = cut_size(adj_matrix, 100)
            print(cut)
        except Exception as e:
            print(e)
