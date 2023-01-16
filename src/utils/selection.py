from trevisan.algorithms import trevisan_fitness
import numpy as np


def pair_wise_selection(population, best_individual, current_generation, adj_matrix, adj_list, active_verts):

    for individual in population:
        trial_cut_value, trial_partition, last_significant_gene = trevisan_fitness(adj_matrix=adj_matrix,
                                                                                   adj_list=adj_list,
                                                                                   active_verts=active_verts,
                                                                                   list_t=individual.trial_gene,
                                                                                   depth_lim=len(individual.trial_gene))

        if trial_cut_value > individual.fitness:
            individual.vector_gene = individual.trial_gene
            individual.fitness = trial_cut_value
            individual.partition = trial_partition
            individual.last_significant_gene = last_significant_gene
            individual.evolution_generation = current_generation

        individual.mutant_gene = np.array([])
        individual.trial_gene = np.array([])

    best_generation = max(population, key=lambda ind: ind.fitness)
    if best_generation.id != best_individual.id:
        best_individual = best_generation
    return best_individual