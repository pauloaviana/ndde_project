import numpy as np


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