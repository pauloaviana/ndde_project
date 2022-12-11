import numpy as np
from shared.population import NovelTrevisanIndividual
from shared.mutation import *
from trevisan_utils import *
from trevisan.trevisan_functions import trevisan_fitness

MUTATION_FUNCTIONS = {"rand/one": de_rand_one, "best/two": de_best_two, "tr_mut": de_trigonometric_mutation}


class NovelTrevisanDE:

    def __init__(self,
                 max_fit: int = 10000,
                 vertices_num: int = 1,
                 active_verts = np.array([]),
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
        self.active_verts = active_verts
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
        self.best_generation = 0
        self.best_individual = max(self.population, key=lambda individual: individual.fitness)


    def start_population(self):
        random_t_matrix = np.random.uniform(0, 1, (self.population_size, 10))
        for list_t in random_t_matrix:

            cut_val, t_partition = trevisan_fitness(adj_matrix = self.adj_matrix,
                                                  adj_list = self.adj_list,
                                                  active_verts = self.active_verts,
                                                  list_t = list_t,
                                                  depth_lim=len(list_t))
            self.population.append(NovelTrevisanIndividual(gene=list_t, partition=t_partition, fitness_value=cut_val))
        return self.population

    def evolutionary_process(self):
        for i in range(self.number_generations):
            if self.best_generation != self.best_individual.evolution_generation:
                best_individual = self.best_individual
                list_t = best_individual.vector_gene
                y = best_individual.partition
                k = best_individual.evolution_generation
                cut_val = best_individual.fitness

                self.best_generation = k

                print(f"Generation {k} | t = {','.join(str(t) for t in list_t)}")
                print(f"Cut Val = {cut_val}")
                print(f"Partition = {y}")
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

    def pairwise_selection(self):
        for individual in self.population:
            trial_cut_value, trial_partition = trevisan_fitness(adj_matrix=self.adj_matrix,
                                                    adj_list=self.adj_list,
                                                    active_verts=self.active_verts,
                                                    list_t=individual.trial_gene,
                                                    depth_lim=len(individual.trial_gene))

            self.fitness_call += 1

            if trial_cut_value > individual.fitness:
                individual.vector_gene = individual.trial_gene
                individual.fitness = trial_cut_value
                individual.partition = trial_partition
                individual.evolution_generation = self.current_generation

            #cleaning mutant and trial vectors for next generation
            individual.mutant_gene = np.array([])
            individual.trial_gene = np.array([])

        self.best_individual = max(self.population, key=lambda ind: ind.fitness)

    def calculate_success_rate(self):
        best_func = ""
        best_rate = 0
        for key, value in self.mutation_status.items():
            success_rate = value[1] / value[0]
            if success_rate > best_rate:
                best_rate = success_rate
                self.current_mutation_method = MUTATION_FUNCTIONS[key]

