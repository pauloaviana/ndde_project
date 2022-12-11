import numpy as np


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
    ev_dic = {}
    for i in range(len(eval)):
        ev_dic[eval[i]] = evec[i]
    min_key = min(ev_dic.keys())
    return ev_dic[min_key]


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


def calculate_fitness_parameters(y, adj_matrix, adj_list):
    n = len(adj_matrix)
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


def find_smalles_eigenvector(adj_matrix, n):
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
    return x


def trevisan_cut(active_verts, n, y, log=True):
    L = [i for i in range(n) if (y[i] == -1) and (i in active_verts)]
    R = [i for i in range(n) if (y[i] == 1) and (i in active_verts)]
    V_prime = [i for i in range(n) if (y[i] == 0) and (i in active_verts)]
    if log:
        print(f"L = {L}")
        print(f"R = {R}")
        print(f"V_prime = {V_prime}")
    return L, R, V_prime