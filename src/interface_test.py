import networkx as nx
import numpy as np
import pandas as pd
import time

from src.utils.files import *
from src.trevisan.algorithms import *
from src.models.differential_evolution_model import DEModel
from src.utils.selection import pair_wise_selection, max_cut_selection
from src.utils.mutation import de_best_two_trevisan, mc_de_rand_one
from src.utils.crossover import exponential_crossover, mc_binomial_crossover
from src.utils.start_population import random_novel_trevisan_init, binary_max_cut_init
from src.utils.ending import limit_generations
from src.utils.graph import create_graph


def main_max_cut_evolutionary(graph):
    adj_matrix = nx.adjacency_matrix(graph).toarray()
    adj_list = get_adj_list(adj_matrix)

    n = len(adj_matrix)

    num_gen = 200
    mut_par = 1.0
    cross_rate = 0.5
    pop_size = 50

    fitness_list = []

    for i in range(1):

        de = DEModel(
            start=binary_max_cut_init,
            mutation=mc_de_rand_one,
            crossover=mc_binomial_crossover,
            selection=max_cut_selection,
            stop_condition=limit_generations,

            population_size=pop_size,
            problem_size=n,
            mutation_rate_f=mut_par,
            max_generation=num_gen,
            adj_matrix=adj_matrix,
            adj_list=adj_list,
            cross_rate=cross_rate
        )

        de.evolutionary_process()
        #print(f"Best Fitness History {de.fitness_history}")
        #print(f"Median Fitness History {de.fitness_avg_history}")
        best_ind = de.best_individual
        R = [i for i in range(len(best_ind.integer_gene)) if best_ind.integer_gene[i] == 0]
        L = [i for i in range(len(best_ind.integer_gene)) if best_ind.integer_gene[i] == 1]
        #print(f"Final fitness {best_ind.fitness}")
        #print(f"R Partition {R}")
        #print(f"L Partition {L}")

        fitness_list.append(best_ind.fitness)

    median_fitness = np.median(np.array(fitness_list), axis=0)
    #print(f"====== Median Fitness: {median_fitness} ======== ")

    return median_fitness, fitness_list


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

df = pd.DataFrame()
run_time = []
cuts = []
median_f = []
graph_name = []
num_edges = []
num_vertices = []

if __name__ == '__main__':
    path = "..\data\erdos_renyi"
    files = list_files(path)



    for file in files:
        graph = create_graph(path, file)
        print(f"[Working with Graph: {file}]\n")

        start_cpu_time = time.process_time()
        median_fit, fitness_list = main_max_cut_evolutionary(graph)
        end_cpu_time = time.process_time()

        clock = end_cpu_time - start_cpu_time
        run_time.append(clock)
        print(f"\nTime = {clock}")

        cuts.append(median_fit)
        print(f"Number of Cuts = {median_fit}")

        median_f.append(fitness_list)

        graph_name.append(file)
        print(f"graph = {file}")

        num_vertices.append(graph.number_of_nodes())
        print(f"Number of Vertices = {graph.number_of_nodes()}")

        num_edges.append(graph.number_of_edges())
        print(f"Number of Edges = {graph.number_of_edges()}\n")

    df['graph'] = graph_name
    df['cuts'] = cuts
    #df['pop_median'] = median_f
    df['vertices'] = num_vertices
    df['edges'] = num_edges
    df['time'] = run_time

    df.head()

    df.to_csv("../statistics/csv_files/DEmodel_m1_c5_p50_.csv")