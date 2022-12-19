from os import walk
import pandas as pd
from trevisan_functions import *
from novel_trevisan_de import NovelTrevisanDE
import time

df = pd.DataFrame()
run_time = []
cuts = []
best_part = []
gens = []
graph_name = []
num_edges = []
num_vertices = []
depth = []

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

    #print(f"Vertices = {active_vertices}")
    num_gen = 55
    mut_par = 0.70
    pop_size = 6

    de = NovelTrevisanDE(vertices_num=n,
                         active_verts=active_vertices,
                         adj_matrix=adj_matrix,
                         adj_list=adj_list,
                         min_aigenvector=x,
                         population_size = pop_size,
                         problem_size = 10,
                         mutation_parameter = mut_par,
                         number_generations = num_gen)

    y, cut_val, k, lsg, = de.evolutionary_process()
    return y, cut_val, k, lsg


if __name__ == '__main__':
    path = "libs"
    files = list_files(path)
    #print(files[0])
    for i in files:
        graph = create_graph(path, i)
        print(f"[Working with Graph: {i}]\n")
        #graph = create_graph(path, 'g05_60_0.csv')
        start_gpu_time = time.process_time()
        y, cut_val, k, lsg = main_evolutionary(graph)
        end_gpu_time = time.process_time()
        num_gen = 55
        mut_par = 0.70
        pop_size = 6

        clock = end_gpu_time - start_gpu_time
        run_time.append(clock)
        print(f"\nTime = {clock}")

        cuts.append(cut_val)
        print(f"Number of Cuts = {cut_val}")

        best_part.append(y)
        print(f"Best Partition = {y}")

        gens.append(k)
        print(f"Generations = {k}")

        depth.append(lsg)
        print(f"Depth = {lsg}")

        graph_name.append(i)
        print(f"graph = {i}")

        num_vertices.append(graph.number_of_nodes())
        print(f"Number of Vertices = {graph.number_of_nodes()}")

        num_edges.append(graph.number_of_edges())
        print(f"Number of Edges = {graph.number_of_edges()}\n")

    df['graph'] = graph_name
    df['cuts'] = cuts
    df['best_partition'] = best_part
    df['generations'] = gens
    df['vertices'] = num_vertices
    df['edges'] = num_edges
    df['time'] = run_time
    df['depth'] = depth
    df['g_max'] = num_gen
    df['mutation'] = mut_par
    df['population'] = pop_size

    df.head()

    df.to_pickle("data_26.pkl")
