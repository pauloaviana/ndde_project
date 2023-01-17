import networkx as nx
from random import randint
from itertools import product


def create_file(graph, path, filename):
    nodes = list(graph.nodes)
    edges = list(graph.edges)
    weights = [randint(1, 10) for i in range(len(edges))]

    with open(f"{path}/{filename}", "w") as file:
        line = f"{len(nodes)},{len(edges)},{sum(weights)}\n"
        file.write(line)
        for i in range(len(edges)):
            edge = edges[i]
            line = f"{edge[0]},{edge[1]},{weights[i]}\n"
            file.write(line)


if __name__ == '__main__':
    path = "../data/erdos_renyi"
    sizes = [40, 80, 120, 160, 200]
    probability = [x/10 for x in range(6, 11)]
    for size, prob in product(sizes, probability):
        for i in range(4):
            graph = nx.erdos_renyi_graph(size, prob)
            filename = f"er_{int(prob*100)}_{size}_{i}.csv"
            create_file(graph, path, filename)