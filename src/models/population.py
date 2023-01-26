from uuid import uuid4
from numpy import array, sin, cos, pi, append


class Individual:

    def __init__(self, discrete_gene, real_gene, fitness_value):
        self.id = uuid4()
        self.discrete_gene = discrete_gene
        self.real_gene = real_gene
        self.mutant_gene = array([])
        self.trial_gene = array([])
        self.fitness = fitness_value
        self.age = 0
        self.last_mutation_method = ""

    def set_mutation_method(self, mutation_method):
        self.last_mutation_method = mutation_method

    def aging(self):
        self.age += 1
        return self

    def gene_to_string(self):
        return "-".join(map(str, self.discrete_gene))

    def to_string(self):
        return f"Gene {self.gene_to_string()}, Fitness = {self.fitness}"


class MaxCutIndividual:

    def __init__(self, real_gene=None):
        self.id = uuid4()

        self.real_gene = real_gene
        self.map_real_to_integer()

    def map_real_to_integer(self):
        self.integer_gene = [0 if bit < 0.5 else 1 for bit in self.real_gene]

    def update_gene(self, new_gene):
        self.real_gene = new_gene
        self.map_real_to_integer()


class TrevisanIndividual:

    def __init__(self, real_gene, partition, fitness_value):
        self.id = uuid4()

        self.real_gene = real_gene
        self.vector_gene = array([cos(real_gene * 2 * pi), sin(real_gene * 2 * pi)])
        self.partition = partition

        self.mutant_gene = array([])
        self.trial_gene = array([])
        self.trial_real_gene = 0.0

        self.fitness = fitness_value
        self.last_mutation_method = ""
        self.evolution_generation = 1


    def to_string(self):
        return f"Gene {self.real_gene:0.5f}, Fitness = {self.fitness}"


class NovelTrevisanIndividual:

    def __init__(self, gene, partition_L, partition_R, partition_V, fitness_value, last_significant_gene):
        self.id = uuid4()
        self.fitness_history = []

        self.vector_gene = gene

        self.partition_L = partition_L
        self.partition_R = partition_R
        self.partition_V = partition_V

        self.mutant_gene = array([])
        self.trial_gene = array([])

        self.fitness = fitness_value
        self.evolution_generation = 1
        self.last_significant_gene = last_significant_gene

        self.complete_gene = self.vector_gene

    def append_gene(self, gene):
        self.complete_gene = append(self.complete_gene, gene)

    def append_partition(self, R, L, V_prime):
        self.partition_R.extend(R)
        self.partition_L.extend(L)
        self.partition_V = V_prime

    def append_history(self, history):
        self.fitness_history.extend(history)

    def increase_fitness(self, fitness):
        self.fitness = fitness

    def to_string(self):
        return f"Gene {self.vector_gene}, Fitness = {self.fitness}"