from uuid import uuid4
from numpy import array, sin, cos, pi


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
        self.age = 0
        self.last_mutation_method = ""
        self.evolution_generation = 1

    def set_mutation_method(self, mutation_method):
        self.last_mutation_method = mutation_method

    def aging(self):
        self.age += 1
        return self

    def gene_to_string(self):
        return "-".join(map(str, self.discrete_gene))

    def to_string(self):
        return f"Gene {self.gene_to_string()}, Fitness = {self.fitness}"