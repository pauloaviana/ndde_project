import random as rn
import numpy as np
from src.models.trevisan_de import TrevisanDE
from src.trevisan.functions import *
from itertools import product


DEPTH_LIM = 50


def trevisan_fitness(adj_matrix, adj_list, active_verts, list_t, depth=0, depth_lim=1, older_R_list=[], older_L_list=[]):

    cut_val, R, V_prime, last_significant_depth = __recursive_trevisan_fitness(adj_matrix=adj_matrix,
                                                                      adj_list=adj_list,
                                                                      active_verts=active_verts,
                                                                      list_t=list_t,
                                                                      depth=depth + 1, depth_lim=depth_lim)

    if len(list_t) < last_significant_depth:
        last_significant_depth = len(list_t)

    list_t = list_t[~np.isnan(list_t)]
    new_ts = []
    for i in range(len(list_t), depth_lim):
        new_ts.append(rn.uniform(0, 1))

    list_t = np.append(list_t, np.array(new_ts))

    L = [node for node in active_verts if node not in R and node not in V_prime]

    C = 0
    R_list, L_list = [], []
    R_list.extend(older_R_list)
    L_list.extend(older_L_list)
    R_list.extend(R)
    L_list.extend(L)

    for r in R_list:
        edges = adj_list[r]
        for e in edges:
            if not (e in R_list) and not (e in V_prime):
                C += adj_matrix[r][e]

    if C > 2000 or C == 0:
        print('here')

    return C, R, L, V_prime, last_significant_depth, list_t


def __recursive_trevisan_fitness(adj_matrix, adj_list, active_verts, list_t, depth=0, depth_lim=1):
    n = len(adj_matrix)
    x = find_smalles_eigenvector(adj_matrix, n)

    y = partition(x, list_t[depth], n)
    L, R, V_prime = trevisan_cut(active_verts, n, y, log=False)

    if len(L) == 0 and len(R) == 0:
        list_t[depth] = None

    if depth + 1 >= depth_lim or len(V_prime) == 0:
        A = []
        V_prime_final = V_prime
        last_significant_depth = depth + 1
    else:
        cut_val, A, V_prime_final, last_significant_depth = __recursive_trevisan_fitness(
                                                            adj_matrix=induced_subgraph(adj_matrix, V_prime),
                                                            adj_list=adj_list, active_verts=V_prime, list_t=list_t,
                                                            depth=depth + 1, depth_lim=depth_lim)

    AL, AR = [], []
    AL.extend(A)
    AL.extend(L)
    AR.extend(A)
    AR.extend(R)

    cut_val_1 = cut_value(adj_matrix, AL, adj_list)
    cut_val_2 = cut_value(adj_matrix, AR, adj_list)

    if cut_val_1 > cut_val_2:
        return cut_val_1, AL, V_prime_final, last_significant_depth
    else:
        return cut_val_2, AR, V_prime_final, last_significant_depth


def trevisan_de(adj_matrix, adj_list, active_verts, num_iter, depth=0):
    n = len(adj_matrix)
    x = find_smalles_eigenvector(adj_matrix, n)

    de = TrevisanDE(vertices_num=n,
                    adj_matrix=adj_matrix,
                    adj_list=adj_list,
                    min_aigenvector=x,
                    population_size=20,
                    mutation_parameter=0.5,
                    number_generations=num_iter)

    de.evolutionary_process()

    best_individual = de.best_individual
    t = best_individual.real_gene
    y = best_individual.partition
    k = best_individual.evolution_generation
    print(f"Depth {depth} | Generation {k} | t = {t:0.5f}")

    L, R, V_prime = trevisan_cut(active_verts, n, y)

    if depth >= 8 or len(V_prime) == 0:
        A = []
    else:
        cut_val, A = trevisan_de(induced_subgraph(adj_matrix, V_prime), adj_list, V_prime, num_iter, depth + 1)

    AL, AR = [], []
    AL.extend(A)
    AL.extend(L)
    AR.extend(A)
    AR.extend(R)
    # Cut_Val is t
    cut_val_1 = cut_value(adj_matrix, AL, adj_list)
    cut_val_2 = cut_value(adj_matrix, AR, adj_list)

    if cut_val_1 > cut_val_2:
        return cut_val_1, AL
    else:
        return cut_val_2, AR


def trevisan_sato(adj_matrix, adj_list, active_verts, num_iter, depth=0):
    n = len(adj_matrix)
    x = find_smalles_eigenvector(adj_matrix, n)

    current_max = 0
    y = np.zeros(n)
    for k in range(num_iter):
        t = rn.uniform(0, 1)

        y_partial = partition(x, t, n)
        c, xx, m = calculate_fitness_parameters(y_partial, adj_matrix, adj_list)
        fit = c + float(xx / 2) - float(m / 2)

        if fit > current_max:
            current_max = fit
            y = y_partial
            print(f"Depth {depth} | Iteration {k} max fit = {current_max:0.2f} | t = {t:0.5f}")

    L, R, V_prime = trevisan_cut(active_verts, n, y)

    if depth >= DEPTH_LIM or len(V_prime) == 0:
        A = []
    else:
        cut_val, A = trevisan_sato(induced_subgraph(adj_matrix, V_prime), adj_list, V_prime, num_iter, depth + 1)

    AL, AR = [], []
    AL.extend(A)
    AL.extend(L)
    AR.extend(A)
    AR.extend(R)
    # Cut_Val is t
    cut_val_1 = cut_value(adj_matrix, AL, adj_list)
    cut_val_2 = cut_value(adj_matrix, AR, adj_list)

    if cut_val_1 > cut_val_2:
        return cut_val_1, AL
    else:
        return cut_val_2, AR
