import numpy as np
import pandas as pd
import networkx as nx
from src.models.novel_trevisan_de import NovelTrevisanDE
from src.trevisan.algorithms import *
from src.utils.graph import create_graph
from itertools import product
import operator


def evolutionary_process(adj_matrix, adj_list, active_vertices, x, **kwargs):

    n = len(adj_matrix)

    best_list = []
    pop_list = []
    v_size_list = []

    for i in range(1):
        de = NovelTrevisanDE(vertices_num=n,
                             active_verts=active_vertices,
                             adj_matrix=adj_matrix,
                             adj_list=adj_list,
                             min_aigenvector=x,
                             population_size=kwargs['pop_size'],
                             problem_size=kwargs['prob_size'],
                             mutation_parameter=kwargs['mut_par'],
                             crossover_probability=kwargs['cross_par'],
                             number_generations=kwargs['num_gen'])

        de.evolutionary_process()
        best_list.append(de.fitness_best_history)
        pop_list.append(de.fitness_median_history)
        v_size_list.append(de.v_size_history)

    median_best = np.median(np.array(best_list), axis=0)
    median_pop = np.median(np.array(pop_list), axis=0)
    median_v_size = np.median(np.array(v_size_list), axis=0)

    print(median_best)
    print(median_pop)
    print(median_v_size)
    return median_best, median_pop, median_v_size


def grid_search(graph):

    adj_matrix = nx.adjacency_matrix(graph).toarray()
    adj_list = get_adj_list(adj_matrix)
    active_vertices = [i for i in range(len(adj_matrix))]

    n = len(adj_matrix)
    x = find_smalles_eigenvector(adj_matrix, n)

    POPULATION_SIZES = [6, 10, 15, 20, 25]
    GENERATIONS_SIZES = [50, 60, 70, 80, 90]
    CROSSOVER_RATES = [0.2, 0.3, 0.4, 0.5, 0.6]
    MUTATION_RATES = [0.3, 0.6, 0.9, 1.2, 1.5]
    PROBLEM_SIZES = [10, 15, 20, 25, 30]

    def find_best_parameters(parameter_type, key, *args):

        print(f"Testing {parameter_type}...")

        results = {}

        for pop_size, num_gen, cross_par, mut_par, prob_size in list(product(*args)):

            kwargs = {
                'prob_size': prob_size,
                'pop_size': pop_size,
                'num_gen': num_gen,
                'mut_par': mut_par,
                'cross_par': cross_par
            }

            print(f"... Testing {parameter_type} - {kwargs[key]}")
            median_best, median_pop, _ = evolutionary_process(adj_matrix, adj_list, active_vertices, x, **kwargs)
            size = len(median_best)-1
            print(f"{parameter_type} - {kwargs[key]}: | Best median fitness: {median_best[size]} | Population Median fitness: {median_pop[size]}")
            results[kwargs[key]] = median_best[size]


        best = max(results.items(), key=operator.itemgetter(1))[0]
        results[best] = -1
        second_best = max(results.items(), key=operator.itemgetter(1))[0]
        return best, second_best


    best_pop_sizes = find_best_parameters("Population Sizes", 'pop_size', POPULATION_SIZES, GENERATIONS_SIZES[2:3],
                                          CROSSOVER_RATES[2:3], MUTATION_RATES[2:3], PROBLEM_SIZES[2:3])

    best_gen_sizes = find_best_parameters("Generation Sizes", "num_gen", best_pop_sizes[:1], GENERATIONS_SIZES,
                                          CROSSOVER_RATES[2:3], MUTATION_RATES[2:3], PROBLEM_SIZES[2:3])

    best_cross_rates = find_best_parameters("Crossover Rates", "cross_par", best_pop_sizes[:1], best_gen_sizes[:1],
                                           CROSSOVER_RATES, MUTATION_RATES[2:3], PROBLEM_SIZES[2:3])

    best_mutation_rates = find_best_parameters("Mutation Rates", "mut_par", best_pop_sizes[:1], best_gen_sizes[:1],
                                             best_cross_rates[:1], MUTATION_RATES, PROBLEM_SIZES[2:3])

    best_prob_sizes = find_best_parameters("Problem Sizes", "prob_size", best_pop_sizes[:1], best_gen_sizes[:1],
                                               best_cross_rates[:1], best_mutation_rates[:1], PROBLEM_SIZES)


    index_list = ['pop_size', 'num_gen', 'mut_par', 'cross_par', 'prob_size']
    results_df = pd.DataFrame(index_list)
    best_params = [best_pop_sizes, best_gen_sizes, best_cross_rates, best_mutation_rates, best_prob_sizes]
    for pop_size, num_gen, cross_par, mut_par, prob_size in list(product(*best_params)):
        kwargs = {
            'prob_size': prob_size,
            'pop_size': pop_size,
            'num_gen': num_gen,
            'mut_par': mut_par,
            'cross_par': cross_par
        }

        print(f"Testing parameters: {prob_size} | {num_gen} | {cross_par} | {mut_par} | {cross_par}...")

        median_best, median_pop, _ = evolutionary_process(adj_matrix, adj_list, active_vertices, x, **kwargs)

        size = len(median_best) - 1
        print(f"...Results = Best median fitness: {median_best[size]} | Population Median fitness: {median_pop[size]}")

        for i in range(len(median_best)):
            kwargs[str(i)] = median_best[i]

        new_df = pd.DataFrame(kwargs, index=[1])
        new_df = new_df.set_index(index_list)
        results_df = results_df.append(new_df)

    results_df.to_csv("../statistics/csv_files/grid_search_icaro_V2.csv")



def search(graph):
    adj_matrix = nx.adjacency_matrix(graph).toarray()
    adj_list = get_adj_list(adj_matrix)
    active_vertices = [i for i in range(len(adj_matrix))]

    n = len(adj_matrix)
    x = find_smalles_eigenvector(adj_matrix, n)
    index_list = ['pop_size', 'num_gen', 'mut_par', 'cross_par', 'prob_size']
    results_df = pd.DataFrame()

    PROBLEM_SIZES = [15, 20, 25, 30, 35]
    for pop_size in PROBLEM_SIZES:
        kwargs = {
            'prob_size': pop_size,
            'pop_size': 15,
            'num_gen': 800,
            'mut_par': 2.3,
            'cross_par': 0.4
        }

        median_best, median_pop, median_v = evolutionary_process(adj_matrix, adj_list, active_vertices, x, **kwargs)

        size = len(median_best) - 1
        print(f"...Results = Best median fitness: {median_best[size]} | Population Median fitness: {median_pop[size]} | V-prime size: {median_v[size]}")

        for i in range(len(median_best)):
            kwargs[str(i)] = median_best[i]

        new_df = pd.DataFrame(kwargs, index=[1])
        new_df = new_df.set_index(index_list)
        results_df = results_df.append(new_df)
    results_df.to_csv("../statistics/csv_files/search_icaro_V3.csv")


if __name__ == '__main__':
    path = "../data/max_cut"
    file = "g05_60_0.csv"
    graph = create_graph(path, file)
    search(graph)


