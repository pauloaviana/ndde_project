import numpy as np
import pandas as pd
import networkx as nx
import operator

from src.trevisan.algorithms import *
from src.models.differential_evolution_model import DEModel
from src.utils.selection import pair_wise_selection, max_cut_selection
from src.utils.mutation import de_best_two_trevisan, mc_de_rand_one
from src.utils.crossover import exponential_crossover, mc_binomial_crossover
from src.utils.start_population import random_novel_trevisan_init, binary_max_cut_init
from src.utils.ending import limit_generations
from src.utils.graph import create_graph

def main_max_cut_evolutionary(adj_matrix,adj_list,**kwargs):
    n = len(adj_matrix)

    best_list = []
    best_history_list = []
    median_history_list = []
    hamming_distance_list = []

    for i in range(3):

        de = DEModel(
            start=binary_max_cut_init,
            mutation_list=mc_de_rand_one,
            crossover=mc_binomial_crossover,
            selection=max_cut_selection,
            stop_condition=limit_generations,

            population_size=kwargs['pop_size'],
            problem_size=n,
            mutation_rate_f=kwargs['mut_par'],
            max_generation=kwargs['num_gen'],
            adj_matrix=adj_matrix,
            adj_list=adj_list,
            cross_rate=kwargs['cross_par']
        )

        de.evolutionary_process()

        best_ind = de.best_individual # Fitness of the best individual
        best_ind_history = de.fitness_history # History of fitness of best individual
        median_pop_history = de.fitness_avg_history # History of median of population's fitness
        median_pop_hamming_distance = de.diversity_avg_history

        # Appending results to the data matrix:

        best_list.append(best_ind.fitness)
        best_history_list.append(best_ind_history)
        median_history_list.append(median_pop_history)
        hamming_distance_list.append(median_pop_hamming_distance)

    # finding the median and the associated solution
    median_best_fitness = np.median(best_list)
    indx = best_list.index(median_best_fitness)
    median_best_history = best_history_list[indx]
    median_best_pop_history = median_history_list[indx]
    median_best_hd = hamming_distance_list[indx]

    return median_best_fitness, best_list, median_best_history, median_best_pop_history, median_best_hd


def grid_search(graph):
    adj_matrix = nx.adjacency_matrix(graph).toarray()
    adj_list = get_adj_list(adj_matrix)

    POPULATION_SIZES = [20, 30, 40, 50, 60]
    GENERATIONS_SIZES = [60, 70, 80, 90, 100]
    CROSSOVER_RATES = [0.2, 0.3, 0.4, 0.5, 0.6]
    MUTATION_RATES = [0.3, 0.6, 0.9, 1.2, 1.5]

    def find_best_parameters(parameter_type, key, *args):

        print(f"Testing {parameter_type}...")

        results = {}

        for pop_size, num_gen, cross_par, mut_par in list(product(*args)):

            kwargs = {
                'pop_size': pop_size,
                'num_gen': num_gen,
                'mut_par': mut_par,
                'cross_par': cross_par
            }

            print(f"... Testing {parameter_type} - {kwargs[key]}")
            median_best, best_list, median_best_history, median_best_pop_history, median_best_hd = main_max_cut_evolutionary(adj_matrix,adj_list, **kwargs)
            print(f"{parameter_type} - {kwargs[key]}: | Best median fitness: {median_best} | List of best fitness: {best_list}")
            results[kwargs[key]] = median_best


        best = max(results.items(), key=operator.itemgetter(1))[0]
        results[best] = -1
        second_best = max(results.items(), key=operator.itemgetter(1))[0]
        return best, second_best


    best_pop_sizes = find_best_parameters("Population Sizes", 'pop_size', POPULATION_SIZES, GENERATIONS_SIZES[2:3],
                                          CROSSOVER_RATES[2:3], MUTATION_RATES[2:3])

    best_gen_sizes = find_best_parameters("Generation Sizes", "num_gen", best_pop_sizes[:1], GENERATIONS_SIZES,
                                          CROSSOVER_RATES[2:3], MUTATION_RATES[2:3])

    best_cross_rates = find_best_parameters("Crossover Rates", "cross_par", best_pop_sizes[:1], best_gen_sizes[:1],
                                           CROSSOVER_RATES, MUTATION_RATES[2:3])

    best_mutation_rates = find_best_parameters("Mutation Rates", "mut_par", best_pop_sizes[:1], best_gen_sizes[:1],
                                             best_cross_rates[:1], MUTATION_RATES)


    #index_list = ['pop_size', 'num_gen', 'mut_par', 'cross_par']
    #results_df = pd.DataFrame(index_list)
    results_df = pd.DataFrame()
    best_params = [best_pop_sizes, best_gen_sizes, best_cross_rates, best_mutation_rates]

    clmn_0 = []
    clmn_1 = []
    clmn_2 = []
    clmn_3 = []
    clmn_4 = []
    clmn_5 = []


    for pop_size, num_gen, cross_par, mut_par in list(product(*best_params)):
        kwargs = {
            'pop_size': pop_size,
            'num_gen': num_gen,
            'mut_par': mut_par,
            'cross_par': cross_par
        }

        print(f"Testing parameters: | {num_gen} | {cross_par} | {mut_par} | {cross_par} |")

        median_best, best_list, median_best_history, median_best_pop_history, median_best_hd = main_max_cut_evolutionary(adj_matrix, adj_list, **kwargs)
        print(f"...Results = Best median fitness: {median_best} | List of best fitness: {best_list}")

        print('median best: ', median_best)
        print('best list: ', best_list)
        print('median_best_history: ', median_best_history)
        print('median_best_hd_history: ', median_best_hd)
        print('median_best_pop_history: ', median_best_pop_history)

        p_list = [pop_size, num_gen, cross_par, mut_par]
        clmn_0.append(p_list)
        clmn_1.append(median_best)
        clmn_2.append(best_list)
        clmn_3.append(median_best_history)
        clmn_4.append(median_best_pop_history)
        clmn_5.append(median_best_hd)

        '''for i in range(len(median_best)):
            kwargs[str(i)] = median_best[i]'''

        '''new_df = pd.DataFrame(kwargs, index=[1])
        new_df = new_df.set_index(index_list)
        results_df = results_df.append(new_df)'''

    results_df['parameters'] = clmn_0
    results_df['median best'] = clmn_1
    results_df['best list'] = clmn_2
    results_df['median history'] = clmn_3
    results_df['median pop history'] = clmn_4
    results_df['median hd history'] = clmn_5

    results_df.to_csv("../statistics/csv_files/grid_DEModelv2.1_mxct_60_0.csv")

if __name__ == '__main__':
    path = "../data/max_cut"
    file = "g05_60_0.csv"
    graph = create_graph(path, file)
    grid_search(graph)


