import numpy as np
import scipy as sp
import networkx as nx
import random as rn


def partition(x, t, n):
    p = np.zeros(n)
    for i in range(n):
        if x[i] <= -np.sqrt(t):
            p[i] = -1
        elif x[i] >= np.sqrt(t):
            p[i] = 1
    return p


def adj_matrix_deg_matrix(M, n):
    deg_matrix = np.zeros([n, n])
    for i in range(n):
        linha = M[i][:]
        soma = np.sum(linha)
        deg_matrix[i][i] = soma
    return deg_matrix


def smallest_eigenvector(M):
    eval, evec = np.linalg.eig(M)
    ev_list = list(zip(eval, evec))
    sorted_list = list(ev_list)
    vect = sorted_list[0][1]
    return vect


def induced_subgraph(adj_matrix, V_prime):
    n = len(adj_matrix)
    sub_graph = np.zeros([n, n])
    for i in range(len(V_prime)):
        for j in range(len(V_prime)):
            sub_graph[i][j] = adj_matrix[i][j]
    return sub_graph


def cut_value(adj_matrix, left_side_vertices, adj_list):
    summation = 0
    for vertex in left_side_vertices:
        for adj in adj_list[vertex]:
            if not (adj in left_side_vertices):
                summation += adj_matrix[vertex][adj]
    return summation


def get_adj_list(adj_matrix):
    adj_list = []
    for i in range(len(adj_matrix)):
        sub_list = []
        for j in range(len(adj_matrix[i])):
            if adj_matrix[i][j] == 1:
                sub_list.append(j)
        adj_list.append(sub_list)
    return adj_list


def calculate_fitness_parameters(y, n):
    m1 = 0
    m2 = 0
    c = 0
    xx = 0

    for i in range(n):
        for j in adj_list[i]:
            if j > i:
                m1 += adj_matrix[i][j]
                if y[i] == 0 and y[j] == 0:
                    m2 += adj_matrix[i][j]
                elif y[i] * y[j] == -1:
                    c += adj_matrix[i][j]
                elif abs(y[i] + y[j]) == 1:
                    xx += adj_matrix[i][j]
    return c, xx, m1-m2


def trevisan(adj_matrix_param, adj_list, active_verts_param, num_iter, depth = 0):
    adj_matrix = adj_matrix_param
    active_verts = active_verts_param

    count = 0

    n = len(adj_matrix)
    D = adj_matrix_deg_matrix(adj_matrix, n)
    neg_sqrt_D = np.zeros([n, n])
    for i in range(n):
        if D[i][i] == 0:
            neg_sqrt_D[i][i] = 0
        else:
            neg_sqrt_D[i, i] = D[i, i] ** (-1 / 2)
    norm_A = neg_sqrt_D @ adj_matrix @ neg_sqrt_D
    x = smallest_eigenvector(norm_A)
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
        c, xx, m = calculate_fitness_parameters(y_partial, n)
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

    if depth >= 100 or len(V_prime) == 0:
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
    graph = nx.Graph()

    rng = np.random.default_rng()
    numbers = rng.choice(10, size=(100, 2), replace=True)
    graph.add_edges_from(numbers)

    adj_matrix = nx.adjacency_matrix(graph).toarray()
    adj_list = get_adj_list(adj_matrix)
    active_vertices = [i for i in range(len(adj_matrix))]
    num_iter = 500

    print(f"Vertices = {active_vertices}")
    cut_val, A = trevisan(adj_matrix, adj_list, active_vertices, num_iter)
    print(f"Cut Val = {cut_val}")
    print(f"Partition = {A}")
