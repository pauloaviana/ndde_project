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
                 problem_size: int = 10,
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
        self.problem_size = problem_size
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
        random_t_matrix = np.random.uniform(0, 1, (self.population_size, self.problem_size))
        for list_t in random_t_matrix:
            cut_val, t_partition, last_significant_gene = trevisan_fitness(adj_matrix = self.adj_matrix,
                                                  adj_list = self.adj_list,
                                                  active_verts = self.active_verts,
                                                  list_t = list_t,
                                                  depth_lim=len(list_t))
            self.population.append(NovelTrevisanIndividual(gene=list_t, partition=t_partition,
                                                           fitness_value=cut_val,
                                                           last_significant_gene=last_significant_gene))
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

    def mutation(self):
        de_best_two_trevisan(self.population, self.best_individual, self.mutation_parameter)


    def exponential_crossover(self):
        ##TODO develop the correct code for crossover
        for individual in self.population:
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


    def pairwise_selection(self):
        for individual in self.population:
            trial_cut_value, trial_partition, last_significant_gene = trevisan_fitness(adj_matrix=self.adj_matrix,
                                                    adj_list=self.adj_list,
                                                    active_verts=self.active_verts,
                                                    list_t=individual.trial_gene,
                                                    depth_lim=len(individual.trial_gene))

            self.fitness_call += 1

            if trial_cut_value > individual.fitness:
                individual.vector_gene = individual.trial_gene
                individual.fitness = trial_cut_value
                individual.partition = trial_partition
                individual.last_significant_gene = last_significant_gene
                individual.evolution_generation = self.current_generation

            #cleaning mutant and trial vectors for next generation
            individual.mutant_gene = np.array([])
            individual.trial_gene = np.array([])

        best_generation = max(self.population, key=lambda ind: ind.fitness)
        if best_generation.id != self.best_individual.id:
            self.best_individual = best_generation
