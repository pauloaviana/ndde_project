import tsplib95
from src.novelDE.graph import Graph
from src.novelDE.novel_de import NovelDiscreteDE
from time import time


if __name__ == '__main__':
    start = time()
    problem = tsplib95.load('../libs/tsp/a280.tsp')

    population_size = 50
    graph = Graph(problem)

    ndde = NovelDiscreteDE(graph, population_size=population_size)
    ndde.evolutionary_process()
    #for individual in ndde.population:
    #    print(individual.to_string())

    print("Best Individual is")
    print(ndde.best_individual.to_string())

    end = time()
    print("Population Generated --- %s seconds ---" % (end - start))
