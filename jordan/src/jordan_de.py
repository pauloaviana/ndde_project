from src.individual import Jordan_Individual
from src.jordan_de_functions import *


import numpy as np

class jordan_de:
    def __init__(self, graph_matrix,
                 graph_coordinates, #list of tuples
                 max_gen: int = 10000,
                 population_size: int = 100,
                 mutation_parameter: float = 0.5,
                 crossover_probability: float = 0.5,
                 polygon_size: int = 10):
        
        self.graph_matrix = graph_matrix
        self.graph_coordinates = graph_coordinates
        self.max_gen = max_gen
        self.population_size = population_size
        self.mutation_parameter = mutation_parameter
        self.crossover_probability = crossover_probability
        self.polygon_size = polygon_size

        self.population = [] #this will turn into a list of Jordan_Individuals
        self.fitness_call = 0
        self.generation = 0
        self.best_individual = max(self.population, key=lambda individual: individual.fitness)

        self.start_population()

    def start_population(self): #get polygons which nodes are evenly distributed around the graph region
        for i in range(self.population_size):
            polygon_coord = np.random.uniform(0,500,2*self.polygon_size) #this returns a numpy array
            #random.uniform bounds must be updated to reflect the scale of the graph coordinates
            partition = jordan_partition(self.graph_coordinates,polygon_coord)
            fit_val = maxcut_evaluation(self.graph_matrix,partition)
            ind = Jordan_Individual(polygon_gene=polygon_coord, associated_partition=partition,fitness_value=fit_val)
            self.population.append(ind)
    
    def evolutionary_process(self):
        for i in range(self.max_gen):
            self.mutation_operator()
            self.crossover_operator()
            self.selection_operator()
            self.generation += 1
    
    def mutation_operator(self):
        rand_one(self.population,self.mutation_parameter)

    def crossover_operator(self):
        exponential_crossover(self.population,self.crossover_probability)

    def selection_operator(self):
        pairwise_selection(self.population,self.graph_matrix,self.graph_coordinates)
