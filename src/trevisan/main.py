from novel_trevisan_de import NovelTrevisanDE
from trevisan_utils import *
from trevisan_functions import *


def main(method):
    # graph = nx.Graph()
    graph = nx.complete_graph(100)
    rng = np.random.default_rng()
    # numbers = rng.choice(10, size=(100, 2), replace=True)
    # numbers = [[0,1],[0,2],[0,3],[1,4],[1,5],[2,6],[2,7],[3,8],[3,9],[4,10],[4,11],[5,12],[5,13]]
    # graph.add_edges_from(numbers)

    adj_matrix = nx.adjacency_matrix(graph).toarray()
    adj_list = get_adj_list(adj_matrix)
    active_vertices = [i for i in range(len(adj_matrix))]
    num_iter = 100

    print(f"Vertices = {active_vertices}")
    cut_val, A = method(adj_matrix, adj_list, active_vertices, num_iter)
    print(f"Cut Val = {cut_val}")
    print(f"Partition = {A}")


def main_evolutionary():
    graph = nx.Graph()
    #graph = nx.complete_graph(40)
    rng = np.random.default_rng()
    #numbers = rng.choice(10, size=(100, 2), replace=True)
    numbers = [[0,1],[0,2],[0,3],[1,4],[1,5],[2,6],[2,7],[3,8],[3,9],[4,10],[4,11],[5,12],[5,13]]
    graph.add_edges_from(numbers)

    adj_matrix = nx.adjacency_matrix(graph).toarray()
    adj_list = get_adj_list(adj_matrix)
    active_vertices = [i for i in range(len(adj_matrix))]
    num_iter = 20

    n = len(adj_matrix)
    x = find_smalles_eigenvector(adj_matrix, n)

    print(f"Vertices = {active_vertices}")

    de = NovelTrevisanDE(vertices_num=n,
                         active_verts=active_vertices,
                         adj_matrix=adj_matrix,
                         adj_list=adj_list,
                         min_aigenvector=x,
                         population_size = 20,
                         mutation_parameter = 0.5,
                         number_generations = num_iter)

    de.evolutionary_process()

if __name__ == '__main__':
    #main(trevisan_de)
    #main(trevisan_random)
    main_evolutionary()
