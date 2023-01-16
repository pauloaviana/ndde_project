from uuid import uuid4

class Jordan_Individual:

    def __init__(self, polygon_gene, associated_partition, fitness_value):
        self.id = uuid4()
        self.polygon_gene = polygon_gene
        self.associated_partition = associated_partition
        self.fitness = fitness_value
        self.mutant_gene = []
        self.trial_polygon = []
        self.age = 0

    def aging(self):
        self.age += 1
        return self
