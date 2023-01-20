import pandas as pd
import networkx as nx
import tsplib95


def __create_graph_from_tsp_file(path, filename):
    filepath = path + "/" + filename
    problem = tsplib95.load(filepath)
    graph = nx.Graph()
    graph.add_edges_from(problem.get_edges())
    graph.add_nodes_from(problem.get_nodes())
    return graph


def create_graph(path, filename, type='max_cut'):
    if type == 'tsp':
        return __create_graph_from_tsp_file(path, filename)

    filepath = path+"/"+filename
    df = pd.read_csv(filepath, header=0, names=['vertice_A', 'vertice_B', 'weight'])
    graph = nx.Graph()
    edges = []
    for index, row in df.iterrows():
        edge = [row['vertice_A'], row['vertice_B']]
        edges.append(edge)
    graph.add_edges_from(edges)
    return graph