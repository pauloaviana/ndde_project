from src.models.population import MaxCutIndividual
from src.utils.fitness import max_cut_fitness
import numpy as np


def mc_binomial_crossover(population, mutant_population, cross_rate, adj_matrix, adj_list):

    trial_population = []
    pop_size = len(population)

    for i in range(pop_size):
        target_vector = population[i].real_gene
        mutant_vector = mutant_population[i].real_gene

        dimension = len(target_vector)

        trial_vector = [-1 for i in range(dimension)]
        random_bit = np.random.randint(0, dimension)

        for j in range(dimension):
            random_value = np.random.randint(0, 1)
            if random_value <= cross_rate or j == random_bit:
                trial_vector[j] = mutant_vector[j]
            else:
                trial_vector[j] = target_vector[j]

        trial_individual = MaxCutIndividual(np.array(trial_vector))
        trial_individual.fitness = max_cut_fitness(trial_individual.integer_gene, adj_matrix, adj_list)
        trial_population.append(trial_individual)

    return trial_population


def exponential_crossover(population):

    for individual in population:
        random_cut = np.random.randint(0, individual.last_significant_gene)
        trial_vector = []
        for i in range(0, random_cut):
            rand = np.random.randint(0, 2)
            if rand == 0:
                trial_vector.append(individual.vector_gene[i])
            elif rand == 1:
                trial_vector.append(individual.mutant_gene[i])

        trial_vector.extend(individual.vector_gene[random_cut:])
        individual.trial_gene = np.array(trial_vector)

    return population