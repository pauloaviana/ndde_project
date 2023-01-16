from trevisan.algorithms import trevisan_fitness
from trevisan.functions import partition, calculate_fitness_parameters
from models.population import NovelTrevisanIndividual, TrevisanIndividual
import numpy as np


def random_novel_trevisan_init(population_size, problem_size, adj_matrix, adj_list, active_verts):

    random_t_matrix = np.random.uniform(0, 1, (population_size, problem_size))
    population = []
    for list_t in random_t_matrix:
        cut_val, t_partition, last_significant_gene = trevisan_fitness(adj_matrix=adj_matrix,
                                                                       adj_list=adj_list,
                                                                       active_verts=active_verts,
                                                                       list_t=list_t,
                                                                       depth_lim=len(list_t))

        population.append(NovelTrevisanIndividual(gene=list_t, partition=t_partition,
                                                       fitness_value=cut_val,
                                                       last_significant_gene=last_significant_gene))
    return population


def random_trevisan_init(population_size, vertices_num, min_aigenvector, adj_matrix, adj_list):
    random_t_list = np.random.uniform(0, 1, (population_size, 1))
    population = []

    for t in random_t_list:
        t = float(t)
        partial_partition = partition(min_aigenvector, t, vertices_num)
        c, x, m = calculate_fitness_parameters(partial_partition, adj_matrix, adj_list)
        fit = c + float(x / 2) - float(m / 2)
        population.append(TrevisanIndividual(real_gene=t, partition=partial_partition, fitness_value=fit))
    return population