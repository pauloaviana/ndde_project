import tsplib95
from graph import Graph
from novel_de import NovelDiscreteDE
from time import time


if __name__ == '__main__':
    start = time()
    problem = tsplib95.load('../libs/tsp/a280.tsp')

    population_size = 100
    graph = Graph(problem)

    ndde = NovelDiscreteDE(graph, population_size=population_size)
    for individual in ndde.population:
        print(individual.to_string())

    end = time()
    print("--- %s seconds ---" % (end - start))
