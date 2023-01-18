import networkx as nx
from itertools import product
from src.utils.files import create_dataset_file


if __name__ == '__main__':
    path = "../data/erdos_renyi"
    sizes = [40, 80, 120, 160, 200]
    probability = [x/10 for x in range(6, 11)]
    for size, prob in product(sizes, probability):
        for i in range(4):
            graph = nx.erdos_renyi_graph(size, prob)
            filename = f"er_{int(prob*100)}_{size}_{i}.csv"
            create_dataset_file(graph, path, filename)