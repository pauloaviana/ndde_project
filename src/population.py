from uuid import uuid4


class Individual:

    def __init__(self, discrete_gene, real_gene, fitness_value):
        self.id = uuid4()
        self.discrete_gene = discrete_gene
        self.real_gene = real_gene
        self.fitness = fitness_value
        self.age = 0

    def fitness_score(self):
        return self.fitness_function(*self.gene)

    def aging(self):
        self.age += 1
        return self

    def gene_to_string(self):
        return "-".join(map(str, self.discrete_gene))

    def to_string(self):
        return f"Gene {self.gene_to_string()}, Fitness = {self.fitness}"