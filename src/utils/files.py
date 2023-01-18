from random import randint
from os import walk


def list_files(path):
    f = []
    for (dirpath, dirnames, filenames) in walk(path):
        if not filenames:
            continue
        for file in filenames:
            if 'g05' in file or 'er' in file:
                f.append(file)
        break
    return f


def create_dataset_file(graph, path, filename):
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