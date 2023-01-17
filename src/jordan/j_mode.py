import numpy as np
from src.individual import Jordan_Individual

class jordanMODE:
    def __init__(self, graph: Graph,
                 max_fit: int = 10000,
                 population_size: int = 100,
                 mutation_parameter: int = 0.1,
                 crossover_probability: int = 0.1,
                 polygon_size: int = 10):
        
        self.graph = graph
        self.population_size = population_size
        self.max_fit = max_fit
        self.fitness_call = 0
        self.generation = 0
        self.mutation_parameter = mutation_parameter
        self.polygon_size = polygon_size

        self.population = []
        self.start_population()

    def start_population(self):
        #generate a population of polygons in which nodes are evenly distributed around the graph region
        return self.population
    
    def evolutionary_process(self):
        for i in range(200):
            self.de_rand_one()
            self.exponential_crossover()
            self.dominance_selection()
            self.generation += 1

    def exponential_crossover(self):
        for individual in self.population:
            individual.trial_gene = individual.mutant_gene

    def dominance_selection(self):