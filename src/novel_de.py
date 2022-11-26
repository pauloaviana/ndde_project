import numpy as np
from population import Individual
from graph import Graph
from mappers import rank_order_value, backward_mapping


class NovelDiscreteDE:

    def __init__(self, graph: Graph, population_size: int = 100):
        self.graph = graph
        self.problem_size = len(self.graph.nodes)
        self.population_size = population_size
        self.population = []

        self.start_population()
        ##k-means clustering
        self.remap_population()

    def start_population(self):
        real_random_tour = np.random.randn(self.population_size, self.problem_size)
        for tour in real_random_tour:
            discrete_tour = rank_order_value(tour)
            individual_fitness = self.graph.get_tour_distance(discrete_tour)
            individual = Individual(discrete_gene=discrete_tour, real_gene=tour, fitness_value=individual_fitness)
            self.population.append(individual)
        return self.population

    def remap_population(self):
        population_discrete_genes = [individual.discrete_gene for individual in self.population]
        population_new_real_genes = list(map(backward_mapping, population_discrete_genes))
        for i in range(len(self.population)):
            individual = self.population[i]
            individual.real_gene = population_new_real_genes[i]

    def evoluationary_proccess(self):
        pass

    def ensemble_mutation(self):
        pass

    def expotencial_crossover(self):
        pass

    def fitness_reevaluation(self):
        pass

    def pairwise_selection(self):
        pass

    def calculate_success_rate(self):
        pass



