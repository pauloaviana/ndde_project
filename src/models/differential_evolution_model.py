import inspect
import numpy as np


class DEModel:

    def __init__(self,
                 start=None,
                 mutation_list=None,
                 crossover=None,
                 selection=None,
                 stop_condition=None,
                 **kwargs):

        self.params = kwargs
        self.params['population'] = []
        self.params['current_generation'] = 0
        self.params['fitness_call'] = 0
        self.params['mutation_function_list'] = mutation_list

        ##functions
        self.crossover_function = crossover
        self.selection_function = selection
        self.start_population_function = start
        self.stop_condition = stop_condition

        self.start_population()
        self.params['best_individual'] = max(self.population, key=lambda individual: individual.fitness)
        self.fitness_history = [self.params['best_individual'].fitness]
        self.fitness_avg_history = [np.median(np.array([ind.fitness for ind in self.population]))]
        self.diversity_history = []
        self.diversity_avg_history = []

    def __getattr__(self, item):
        return self.params[item]

    def start_population(self):
        start_params = self.__get_params(self.start_population_function)
        self.params['population'] = self.start_population_function(**start_params)

    def evolutionary_process(self):
        while not self.check_stop_condition():
            self.params['current_generation'] += 1
            self.mutation()
            self.crossover()
            self.selection()

    def mutation(self):
        mutant_population = []
        for individual in self.population:
            mutation_function = individual.get_mutation()
            mutation_params = self.__get_params(mutation_function)
            mutant_population.append(mutation_function(**mutation_params))
        self.params['mutant_population'] = mutant_population

    def crossover(self):
        crossover_params = self.__get_params(self.crossover_function)
        self.params['trial_population'] = self.crossover_function(**crossover_params)

    def selection(self):
        selection_params = self.__get_params(self.selection_function)
        self.params['population'] = self.selection_function(**selection_params)
        self.params['best_individual'] = max(self.population, key=lambda individual: individual.fitness)
        self.params['mutant_population'] = []
        self.params['trial_population'] = []
        self.fitness_history.append(self.params['best_individual'].fitness)
        self.fitness_avg_history.append(int(np.median(np.array([ind.fitness for ind in self.population]))))
        self.diversity_history.append(int(self.params['best_individual'].median_hamming_distance))
        self.diversity_avg_history.append(int(np.median(np.array([ind.median_hamming_distance for ind in self.population]))))

    def check_stop_condition(self):
        stop_condition_params = self.__get_params(self.stop_condition)
        return self.stop_condition(**stop_condition_params)

    def __get_params(self, function):
        function_params = {}
        arg_names = inspect.getfullargspec(function)[0]
        for arg in arg_names:
            function_params[arg] = self.params[arg]
        return function_params