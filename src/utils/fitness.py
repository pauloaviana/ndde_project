def max_cut_fitness(integer_gene, adj_matrix, adj_list):
    partition = [i for i in range(len(integer_gene)) if integer_gene[i] == 0]
    fitness = 0

    for ver in partition:
        edges = adj_list[ver]
        for e in edges:
            if not (e in partition):
                fitness += adj_matrix[ver][e]

    return fitness