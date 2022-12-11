import numpy as np
import networkx as nx
import random as rn
from trevisan_de import TrevisanDE
from trevisan_utils import *


def trevisan_de(adj_matrix, adj_list, active_verts, num_iter, depth = 0):

    n = len(adj_matrix)
    D = adj_matrix_deg_matrix(adj_matrix, n)
    neg_sqrt_D = np.zeros([n, n])
    for i in range(n):
        if D[i][i] == 0:
            neg_sqrt_D[i][i] = 0
        else:
            neg_sqrt_D[i, i] = D[i, i] ** (-1 / 2)
    norm_A = neg_sqrt_D @ adj_matrix @ neg_sqrt_D
    x = smallest_eigenvector(norm_A + np.identity(n))
    x_norm = x / max(abs(x))

    current_max = 0
    X = 0  # total weight of the edges between L ∪ R and V'
    M = 0  # total weight of all edges − total weight of the edges between vertices of V ′.
    C = 0  # Number of partial cuts
    final_partition = np.zeros(n)

    ####### A implementação do DE entraria nesse loop
    de = TrevisanDE(vertices_num = n,
                     min_aigenvector = x,
                     population_size = 50,
                     mutation_parameter = 0.5,
                     number_generations = num_iter)

    de.evolutionary_process()

    best_individual = de.best_individual
    t = best_individual.real_gene
    y = best_individual.partition
    k = best_individual.evolution_generation
    print(f"Depth {depth} | Iteration {k} | t = {t:0.5f}")

    L = [i for i in range(n) if (y[i] == -1) and (i in active_verts)]
    R = [i for i in range(n) if (y[i] == 1) and (i in active_verts)]
    V_prime = [i for i in range(n) if (y[i] == 0) and (i in active_verts)]

    print(f"L = {L}")
    print(f"R = {R}")
    print(f"V_prime = {V_prime}")

    if depth >= 8 or len(V_prime) == 0:
        A = []
    else:
        cut_val, A = trevisan_de(induced_subgraph(adj_matrix, V_prime), adj_list, V_prime, num_iter, depth + 1)

    A.extend(L)
    A.extend(R)
    # Cut_Val is t
    cut_val_1 = cut_value(adj_matrix, A, adj_list)
    cut_val_2 = cut_value(adj_matrix, A, adj_list)

    if cut_val_1 > cut_val_2:
        return (cut_val_1, A)
    else:
        return (cut_val_2, A)




def trevisan(adj_matrix_param, adj_list, active_verts_param, num_iter, depth = 0):
    adj_matrix = adj_matrix_param
    active_verts = active_verts_param

    n = len(adj_matrix)
    D = adj_matrix_deg_matrix(adj_matrix, n)
    neg_sqrt_D = np.zeros([n, n])
    for i in range(n):
        if D[i][i] == 0:
            neg_sqrt_D[i][i] = 0
        else:
            neg_sqrt_D[i, i] = D[i, i] ** (-1 / 2)
    norm_A = neg_sqrt_D @ adj_matrix @ neg_sqrt_D
    x = smallest_eigenvector(norm_A + np.identity(n))
    x_norm = x / max(abs(x))

    current_max = 0
    X = 0  # total weight of the edges between L ∪ R and V'
    M = 0  # total weight of all edges − total weight of the edges between vertices of V ′.
    C = 0  # Number of partial cuts
    final_partition = np.zeros(n)

    ####### A implementação do DE entraria nesse loop

    current_max = 0
    y = np.zeros(n)
    for k in range(num_iter):
        t = rn.uniform(0, 1)

        y_partial = partition(x, t, n)
        c, xx, m = calculate_fitness_parameters(y_partial, adj_matrix, adj_list)
        fit = c + float(xx / 2) - float(m / 2)

        if fit > current_max:
            current_max = fit
            M = m
            C = c
            X = xx
            y = y_partial
            print(f"Depth {depth} | Iteration {k} max fit = {current_max:0.2f} | t = {t:0.5f}")

    L = [i for i in range(n) if (y[i] == -1) and (i in active_verts)]
    R = [i for i in range(n) if (y[i] == 1) and (i in active_verts)]
    V_prime = [i for i in range(n) if (y[i] == 0) and (i in active_verts)]

    if depth >= 10 or len(V_prime) == 0:
        A = []
    else:
        cut_val, A = trevisan(induced_subgraph(adj_matrix, V_prime), adj_list, V_prime, num_iter, depth+1)

    A.extend(L)
    A.extend(R)
    # Cut_Val is t
    cut_val_1 = cut_value(adj_matrix, A, adj_list)
    cut_val_2 = cut_value(adj_matrix, A, adj_list)

    if cut_val_1 > cut_val_2:
        return (cut_val_1, A)
    else:
        return (cut_val_2, A)


if __name__ == '__main__':
    #graph = nx.Graph()
    graph = nx.complete_graph(10)
    rng = np.random.default_rng()
    #numbers = rng.choice(10, size=(100, 2), replace=True)
    #numbers = [[0,1],[0,2],[0,3],[1,4],[1,5],[2,6],[2,7],[3,8],[3,9],[4,10],[4,11],[5,12],[5,13]]
    #graph.add_edges_from(numbers)

    adj_matrix = nx.adjacency_matrix(graph).toarray()
    adj_list = get_adj_list(adj_matrix)
    active_vertices = [i for i in range(len(adj_matrix))]
    num_iter = 50

    print(f"Vertices = {active_vertices}")
    cut_val, A = trevisan_de(adj_matrix, adj_list, active_vertices, num_iter)
    print(f"Cut Val = {cut_val}")
    print(f"Partition = {A}")
