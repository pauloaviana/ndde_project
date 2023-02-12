import networkx as nx
import numpy as np
import pandas as pd

from src.trevisan.algorithms import *
from src.models.differential_evolution_model import DEModel
from src.utils.selection import pair_wise_selection, max_cut_selection, max_cut_hamming_selection
from src.utils.mutation import de_best_two_trevisan, mc_de_rand_one
from src.utils.crossover import exponential_crossover, mc_binomial_crossover
from src.utils.start_population import random_novel_trevisan_init, binary_max_cut_init
from src.utils.ending import limit_generations
from src.utils.graph import create_graph


def main_max_cut_evolutionary(graph):
    adj_matrix = nx.adjacency_matrix(graph).toarray()
    adj_list = get_adj_list(adj_matrix)

    n = len(adj_matrix)

    num_gen = 100
    mut_par = 1.0
    cross_rate = 0.3
    pop_size = 50
    diversity_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    dic_results = {}

    for diversity_rate in diversity_rates:
        fitness_list = []

        for i in range(5):
            print(f"====== Diversity Rate {diversity_rate} | Iteration Number {i} ======== ")

            de = DEModel(
                start=binary_max_cut_init,
                mutation=mc_de_rand_one,
                crossover=mc_binomial_crossover,
                selection=max_cut_hamming_selection,
                stop_condition=limit_generations,

                population_size=pop_size,
                problem_size=n,
                mutation_rate_f=mut_par,
                max_generation=num_gen,
                adj_matrix=adj_matrix,
                adj_list=adj_list,
                cross_rate=cross_rate,
                diversity_rate=diversity_rate
            )

            de.evolutionary_process()
            #print(f"Best Fitness History {de.fitness_history}")
            #print(f"Median Fitness History {de.fitness_avg_history}")
            #print(f"Hamming Distance History {de.diversity_history}")
            #print(f"Median Hamming Distance History {de.diversity_avg_history}")
            best_ind = de.best_individual
            R = [i for i in range(len(best_ind.integer_gene)) if best_ind.integer_gene[i] == 0]
            L = [i for i in range(len(best_ind.integer_gene)) if best_ind.integer_gene[i] == 1]
            print(f"Final fitness {best_ind.fitness}")
            #print(f"R Partition {R}")
            #print(f"L Partition {L}")

            fitness_list.append(best_ind.fitness)

        median_fitness = np.median(np.array(fitness_list), axis=0)
        print(f"====== Median Fitness: {median_fitness} ======== ")
        dic_results[diversity_rate] = fitness_list

    df = pd.DataFrame(data=dic_results)
    return df


def main_evolutionary(graph):

    adj_matrix = nx.adjacency_matrix(graph).toarray()
    adj_list = get_adj_list(adj_matrix)
    active_vertices = [i for i in range(len(adj_matrix))]

    n = len(adj_matrix)
    x = find_smalles_eigenvector(adj_matrix, n)

    num_gen = 60
    mut_par = 0.50
    pop_size = 6

    de = DEModel(
                start=random_novel_trevisan_init,
                mutation=de_best_two_trevisan,
                crossover=exponential_crossover,
                selection=pair_wise_selection,
                stop_condition=limit_generations,

                population_size=pop_size,
                mutation_rate_f=mut_par,
                number_generations=num_gen,
                max_generation=50,
                problem_size=10,
                vertices_num=n,
                active_verts=active_vertices,
                adj_matrix=adj_matrix,
                adj_list=adj_list,
                min_aigenvector=x
    )

    de.evolutionary_process()
    best_individual = de.best_individual

    y = best_individual.partition
    k = best_individual.evolution_generation
    cut_val = best_individual.fitness
    lsg = best_individual.last_significant_gene
    num_gen = de.number_generations
    mut_par = de.mutation_rate_f
    pop_size = de.population_size

    return y, cut_val, k, lsg, num_gen, mut_par, pop_size


if __name__ == '__main__':
    path = "..\data\max_cut"
    file = "g05_100_0.csv"
    graph = create_graph(path, file)
    df = main_max_cut_evolutionary(graph)
    df.to_csv("../statistics/csv_files/hamming_icaro.csv")
