import numpy as np

from src.utils.mutation import *
from src.trevisan.functions import *
from src.trevisan.algorithms import trevisan_fitness

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
                 problem_size: int = 50,
                 problem_batch_size: int = 10,
                 mutation_parameter: float = 0.1,
                 crossover_probability: float = 0.1,
                 number_generations: int = 200,
                 mutation_cycle=3):

        self.adj_matrix = adj_matrix
        self.current_adj_matrix = adj_matrix

        self.adj_list = adj_list
        self.active_verts = active_verts
        self.min_aigenvector = min_aigenvector
        self.vertices_num = vertices_num

        self.number_generations = number_generations
        self.population_size = population_size
        self.problem_size = problem_size
        self.problem_batch_size = problem_batch_size
        self.current_batch = 1
        self.max_fit = max_fit
        self.fitness_call = 0
        self.current_generation = 0
        self.mutation_parameter = mutation_parameter
        self.mutation_cycle = mutation_cycle
        self.current_mutation_method = None

        self.mutation_status = {"rand/one": [0, 0], "best/two": [0, 0], "tr_mut": [0, 0]}
        self.crossover_probability = crossover_probability
        self.population = []
        self.batch_best_individual = None

        self.population_best_individual = NovelTrevisanIndividual(gene=np.array([]),
                                                                  partition_R=[],
                                                                  partition_L=[],
                                                                  partition_V=[],
                                                                  fitness_value=0,
                                                                  last_significant_gene=0)
        self.start_population()
        self.best_generation = 0

        self.fitness_best_history = []
        self.fitness_median_history = []
        self.v_size_history = []
        self.update_history()

    def update_history(self):

        median_fitness = np.median(np.array([ind.fitness for ind in self.population]))
        self.fitness_best_history.append(self.batch_best_individual.fitness)
        self.fitness_median_history.append(median_fitness)
        self.v_size_history.append(len(self.batch_best_individual.partition_V))

        print(f"Generation {self.current_generation}")
        print(f"Best Fitness {self.batch_best_individual.fitness}")
        #print(f"Median Fitness {median_fitness}")
        #print(f"Best V Prime Size {len(self.batch_best_individual.partition_V)}")

    def update_population_best_individual(self):

        R = self.batch_best_individual.partition_R
        L = self.batch_best_individual.partition_L
        V = self.batch_best_individual.partition_V

        self.population_best_individual.append_gene(self.batch_best_individual.vector_gene)
        self.population_best_individual.append_history(self.batch_best_individual.fitness_history)
        self.population_best_individual.append_partition(R, L, V)
        self.population_best_individual.increase_fitness(self.batch_best_individual.fitness)

    def start_population(self):

        partition_R = self.population_best_individual.partition_R
        partition_L = self.population_best_individual.partition_L

        random_t_matrix = np.random.uniform(0, 1, (self.population_size, self.problem_batch_size))
        for list_t in random_t_matrix:

            depth_lim = self.problem_batch_size
            cut_val, R, L, V_prime, last_significant_gene, new_list_t = trevisan_fitness(adj_matrix=self.adj_matrix,
                                                                                         adj_list = self.adj_list,
                                                                                         active_verts = self.active_verts,
                                                                                         list_t = list_t,
                                                                                         depth_lim=depth_lim,
                                                                                         older_R_list= partition_R,
                                                                                         older_L_list= partition_L)


            self.population.append(NovelTrevisanIndividual(gene=new_list_t,
                                                           partition_R=R,
                                                           partition_L=L,
                                                           partition_V=V_prime,
                                                           fitness_value=cut_val,
                                                           last_significant_gene=last_significant_gene))

            self.batch_best_individual = max(self.population, key=lambda individual: individual.fitness)

        return self.population

    def check_restart_population(self):
        if self.current_generation % self.problem_batch_size == 0:
            self.current_batch += 1

            self.update_population_best_individual()
            self.population = []
            self.active_verts = self.batch_best_individual.partition_V
            self.start_population()

    def evolutionary_process(self):
        for i in range(self.number_generations):
            self.current_generation += 1
            self.mutation()
            self.exponential_crossover()
            self.pairwise_selection()
            self.check_restart_population()
            self.update_history()

        #TRYING TO SAVE EACH INDIVIDUAL'S FITNESS HISTORY
        pop_fit_history = []
        for individual in self.population:
            pop_fit_history.append(individual.fitness_history)

        #best_individual = self.population_best_individual
        best_individual = self.batch_best_individual

        R = best_individual.partition_R
        L = best_individual.partition_L
        V = best_individual.partition_V

        k = best_individual.evolution_generation

        cut_val = best_individual.fitness
        lsg = best_individual.last_significant_gene

        num_gen = self.number_generations
        mut_par = self.mutation_parameter
        pop_size = self.population_size

        return R, L, V, cut_val, k, lsg, num_gen, mut_par, pop_size, pop_fit_history
        

    def mutation(self):
        de_rand_one_trevisan(self.population, self.mutation_parameter)


    def exponential_crossover(self):
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
        individual.trial_gene = individual.mutant_gene

    def pairwise_selection(self):
        for individual in self.population:

            partition_R = self.population_best_individual.partition_R
            partition_L = self.population_best_individual.partition_L

            trial_cut_value, R, L, V_prime, last_significant_gene, new_list_t = trevisan_fitness(adj_matrix=self.adj_matrix,
                                                                                                 adj_list=self.adj_list,
                                                                                                 active_verts=self.active_verts,
                                                                                                 list_t=individual.trial_gene,
                                                                                                 depth_lim=len(individual.trial_gene),
                                                                                                 older_R_list=partition_R,
                                                                                                 older_L_list=partition_L)

            self.fitness_call += 1

            if trial_cut_value > individual.fitness:
                individual.vector_gene = new_list_t
                individual.fitness = trial_cut_value
                individual.partition_R = R
                individual.partition_L = L
                individual.partition_V = V_prime
                individual.last_significant_gene = last_significant_gene
                individual.evolution_generation = self.current_generation

            #cleaning mutant and trial vectors for next generation
            individual.fitness_history.append(individual.fitness)
            individual.mutant_gene = np.array([])
            individual.trial_gene = np.array([])

        best_generation = max(self.population, key=lambda ind: ind.fitness)
        if best_generation.id != self.batch_best_individual.id:
            self.batch_best_individual = best_generation
