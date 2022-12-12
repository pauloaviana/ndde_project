from os import walk
import pandas as pd
from trevisan_functions import *
from trevisan.novel_trevisan_de import NovelTrevisanDE


def list_files(path):
    f = []
    for (dirpath, dirnames, filenames) in walk(path):
        if not filenames:
            continue
        for file in filenames:
            if 'g05' in file:
                f.append(file)
        break
    return f


def create_graph(path, filename):
    filepath = path+"/"+filename
    df = pd.read_csv(filepath, header=0, names=['vertice_A', 'vertice_B', 'weight'])
    graph = nx.Graph()
    edges = []
    for index, row in df.iterrows():
        edge = [row['vertice_A'], row['vertice_B']]
        edges.append(edge)
    graph.add_edges_from(edges)
    return graph


def main_evolutionary(graph):

    adj_matrix = nx.adjacency_matrix(graph).toarray()
    adj_list = get_adj_list(adj_matrix)
    active_vertices = [i for i in range(len(adj_matrix))]

    n = len(adj_matrix)
    x = find_smalles_eigenvector(adj_matrix, n)

    print(f"Vertices = {active_vertices}")

    de = NovelTrevisanDE(vertices_num=n,
                         active_verts=active_vertices,
                         adj_matrix=adj_matrix,
                         adj_list=adj_list,
                         min_aigenvector=x,
                         population_size = 10,
                         problem_size = 10,
                         mutation_parameter = 0.5,
                         number_generations = 50)

    de.evolutionary_process()


if __name__ == '__main__':
    path = "../../libs/max_cut"
    files = list_files(path)
    print(files[0])
    graph = create_graph(path, 'g05_60_0.csv')
    main_evolutionary(graph)
