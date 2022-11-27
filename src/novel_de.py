import numpy as np
from population import Individual
from graph import Graph
from mappers import rank_order_value, backward_mapping
from mutation import *


MUTATION_FUNCTIONS = {"rand/one": de_rand_one, "best/two": de_best_two, "tr_mut": de_trigonometric_mutation}

class NovelDiscreteDE:

    def __init__(self, graph: Graph,
                 max_fit: int = 10000,
                 population_size: int = 100,
                 mutation_parameter: int = 0.1,
                 crossover_probability: int = 0.1,
                 mutation_cycle=3):

        self.graph = graph
        self.problem_dimension = len(self.graph.nodes)
        self.population_size = population_size
        self.max_fit = max_fit
        self.fitness_call = 0
        self.generation = 0
        self.mutation_parameter = mutation_parameter
        self.mutation_cycle = mutation_cycle
        self.current_mutation_method = None

        self.mutation_status = {"rand/one": [0, 0], "best/two": [0, 0], "tr_mut": [0, 0]}
        self.crossover_probability = crossover_probability
        self.population = []

        self.start_population()
        ##k-means clustering
        self.remap_population()

        self.best_individual = max(self.population, key=lambda individual: individual.fitness)

    def start_population(self):
        real_random_tour = np.random.randn(self.population_size, self.problem_dimension)
        for tour in real_random_tour:
            discrete_tour = rank_order_value(tour)
            individual_fitness = self.graph.get_tour_distance(discrete_tour)
            individual = Individual(discrete_gene=discrete_tour, real_gene=tour, fitness_value=individual_fitness)
            self.population.append(individual)
            self.fitness_call += 1
        return self.population

    def remap_population(self):
        population_discrete_genes = [individual.discrete_gene for individual in self.population]
        population_new_real_genes = list(map(backward_mapping, population_discrete_genes))
        for i in range(len(self.population)):
            individual = self.population[i]
            individual.real_gene = population_new_real_genes[i]

    def evolutionary_process(self):
        for i in range(200):
            self.ensemble_mutation()
            self.exponential_crossover()
            self.pairwise_selection()
            self.calculate_success_rate()
            self.generation += 1

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
            discrete_trial_gene = rank_order_value(individual.trial_gene)
            trial_fitness = self.graph.get_tour_distance(discrete_trial_gene)
            self.fitness_call += 1
            if trial_fitness < individual.fitness:
                individual.real_gene = individual.trial_gene
                individual.discrete_gene = discrete_trial_gene
                individual.fitness = trial_fitness
                if self.generation % self.mutation_cycle == 0:
                    self.mutation_status[individual.last_mutation_method][1] += 1

            #cleaning mutant and trial vectors for next generation
            individual.mutant_gene = np.array([])
            individual.trial_gene = np.array([])
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

