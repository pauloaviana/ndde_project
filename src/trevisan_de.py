import numpy as np
from population import TrevisanIndividual
from mutation import *
from trevisan_utils import *


MUTATION_FUNCTIONS = {"rand/one": de_rand_one, "best/two": de_best_two, "tr_mut": de_trigonometric_mutation}


class TrevisanDE:

    def __init__(self,
                 max_fit: int = 10000,
                 vertices_num: int = 1,
                 adj_matrix = np.array([[]]),
                 adj_list = np.array([[]]),
                 min_aigenvector = np.array([]),
                 population_size: int = 100,
                 mutation_parameter: int = 0.1,
                 crossover_probability: int = 0.1,
                 number_generations: int = 200,
                 mutation_cycle=3):

        self.adj_matrix = adj_matrix
        self.adj_list = adj_list
        self.min_aigenvector = min_aigenvector
        self.vertices_num = vertices_num

        self.number_generations = number_generations
        self.population_size = population_size
        self.max_fit = max_fit
        self.fitness_call = 0
        self.current_generation = 0
        self.mutation_parameter = mutation_parameter
        self.mutation_cycle = mutation_cycle
        self.current_mutation_method = None

        self.mutation_status = {"rand/one": [0, 0], "best/two": [0, 0], "tr_mut": [0, 0]}
        self.crossover_probability = crossover_probability
        self.population = []

        self.start_population()
        self.best_individual = max(self.population, key=lambda individual: individual.fitness)


    def start_population(self):
        random_t_list = np.random.uniform(0, 1, (self.population_size, 1))
        for t in random_t_list:
            t = float(t)
            partial_partition = partition(self.min_aigenvector, t, self.vertices_num)
            c, x, m = calculate_fitness_parameters(partial_partition, self.adj_matrix, self.adj_list)
            fit = c + float(x / 2) - float(m / 2)
            self.population.append(TrevisanIndividual(real_gene=t, partition=partial_partition, fitness_value=fit))
        return self.population

    def evolutionary_process(self):
        for i in range(self.number_generations):
            self.current_generation += 1
            self.mutation()
            self.exponential_crossover()
            self.pairwise_selection()
            #self.calculate_success_rate()

    def mutation(self):
        de_best_two_trevisan(self.population, self.mutation_parameter)

    def ensemble_mutation(self):

        if self.generation % self.mutation_cycle == 0:

            rng = np.random.default_rng()
            random_indexes = rng.choice(self.population_size, size=self.population_size, replace=False)
            break_one = int(np.round(self.population_size/3))

            population_one = [self.population[i] for i in random_indexes[:break_one]]
            population_two = [self.population[i] for i in random_indexes[break_one:break_one*2]]
            population_three = [self.population[i] for i in random_indexes[break_one*2:]]

            self.mutation_status = {
                            "rand/one": [len(population_one), 0],
                            "best/two": [len(population_two), 0],
                            "tr_mut": [len(population_three), 0]
            }

            de_rand_one(population_one, self.mutation_parameter)
            de_best_two(population_two, self.mutation_parameter)
            de_trigonometric_mutation(population_three)

            for ind in population_one:
                ind.set_mutation_method("rand/one")
            for ind in population_two:
                ind.set_mutation_method("best/two")
            for ind in population_three:
                ind.set_mutation_method("tr_mut")

        elif self.current_mutation_method == MUTATION_FUNCTIONS["tr_mut"]:
            self.current_mutation_method(self.population)
        else:
            self.current_mutation_method(self.population, self.mutation_parameter)

    def exponential_crossover(self):
        ##TODO develop the correct code for crossover
        for individual in self.population:
            individual.trial_gene = individual.mutant_gene
            arctan = np.arctan(individual.trial_gene[0] / individual.trial_gene[1])
            gene = abs(arctan) / (np.pi / 2)
            individual.trial_real_gene = gene

    def pairwise_selection(self):
        for individual in self.population:

            trial_partition = partition(self.min_aigenvector, individual.trial_real_gene, self.vertices_num)
            c, x, m = calculate_fitness_parameters(trial_partition, self.adj_matrix, self.adj_list)
            trial_fitness = c + float(x / 2) - float(m / 2)
            self.fitness_call += 1

            if trial_fitness > individual.fitness:
                individual.real_gene = individual.trial_real_gene
                individual.vector_gene = individual.trial_gene
                individual.fitness = trial_fitness
                individual.partition = trial_partition
                individual.evolution_generation = self.current_generation

            #cleaning mutant and trial vectors for next generation
            individual.mutant_gene = np.array([])
            individual.trial_gene = np.array([])
            individual.trial_real_gene = 0.0
            individual.last_mutation_method = ""

        self.best_individual = max(self.population, key=lambda ind: ind.fitness)

    def calculate_success_rate(self):
        best_func = ""
        best_rate = 0
        for key, value in self.mutation_status.items():
            success_rate = value[1] / value[0]
            if success_rate > best_rate:
                best_rate = success_rate
                self.current_mutation_method = MUTATION_FUNCTIONS[key]



