from src.trevisan.algorithms import trevisan_fitness
from src.models.population import MaxCutIndividual
from src.utils.fitness import max_cut_fitness
import numpy as np


def hamming_distance(first_gene: str, second_gene: str) -> int:
    dist_counter = 0
    for n in range(len(first_gene)):
        if first_gene[n] != second_gene[n]:
            dist_counter += 1
    return dist_counter


def __max_cut_filter_redundant_population(new_population, limit_removed):
    filtered_population = []
    for i in range(len(new_population)):
        ind = new_population[i]
        equal_individuals = np.where(ind.list_hamming_distance == 0)[0].tolist()
        equal_individuals = list(filter(lambda x: x > i, equal_individuals)) #take only the next individuals
        filtered_population.extend(equal_individuals)

    size = len(filtered_population)
    if size == 0:
        return new_population
    elif size > limit_removed:
        filtered_population = filtered_population[size-limit_removed:]

    return np.delete(new_population, np.array(filtered_population)).tolist()


def max_cut_hamming_selection(population, trial_population, current_generation, adj_matrix, adj_list, diversity_rate):
    new_population = max_cut_selection(population, trial_population)
    new_population.sort(key=lambda x: x.fitness)

    for ind in new_population:
        distances = np.array([hamming_distance(ind.integer_gene, other.integer_gene) for other in new_population])
        ind.list_hamming_distance = distances
        ind.median_hamming_distance = np.median(distances)

    if current_generation % 10 == 9:
        limit_removed = round(len(new_population) * diversity_rate)
        if limit_removed == 0:
            return new_population

        new_population = __max_cut_filter_redundant_population(new_population, limit_removed)
        old_size, new_size = len(population), len(new_population)
        if old_size != new_size:
            problem_size = len(new_population[0].real_gene)
            for i in range(old_size-new_size):
                real_gene = np.random.uniform(0, 1, problem_size)

                individual = MaxCutIndividual(real_gene)
                individual.fitness = max_cut_fitness(individual.integer_gene, adj_matrix, adj_list)
                new_population.append(individual)

            for ind in new_population:
                distances = np.array([hamming_distance(ind.integer_gene, other.integer_gene) for other in population])
                ind.list_hamming_distance = distances
                ind.median_hamming_distance = np.median(distances)

            avg = int(np.median(np.array([ind.median_hamming_distance for ind in new_population])))

    return new_population


def max_cut_selection(population, trial_population):

    new_population = []

    for i in range(len(population)):
        trial_ind = trial_population[i]
        target_ind = population[i]

        if trial_ind.fitness > target_ind.fitness and trial_ind not in population:
            new_population.append(trial_ind)
        else:
            new_population.append(target_ind)

    return new_population


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