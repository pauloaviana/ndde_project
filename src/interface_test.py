import pandas as pd
import networkx as nx
from src.trevisan.algorithms import *
from src.models.differential_evolution_model import DEModel
from src.utils.selection import pair_wise_selection
from src.utils.mutation import de_best_two_trevisan
from src.utils.crossover import exponential_crossover
from src.utils.start_population import random_novel_trevisan_init
from src.utils.ending import limit_generations


def create_graph(path, filename):
    filepath = path+"/"+filename
    df = pd.read_csv(filepath, header=0, names=['vertice_A', 'vertice_B', 'weight'])
    graph = nx.Graph()
    edges = []
    for index, row in df.iterrows():
        edge = [row['vertice_A'], row['vertice_B']]
        edges.append(edge)
    graph.add_edges_from(edges)
    return graph


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
    file = "g05_60_0.csv"
    graph = create_graph(path, file)
    main_evolutionary(graph)
