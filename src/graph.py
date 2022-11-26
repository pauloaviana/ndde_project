from tsplib95.models import StandardProblem


class Graph:

    def __init__(self, problem: StandardProblem):
        self.problem = problem
        self.nodes = {}
        self.edges = {}
        self.__create_nodes()
        self.__create_edges()

    def get_tour_distance(self, tour):
        total_distance = 0
        for i in range(len(tour)-1):
            edge = (tour[i], tour[i+1])
            total_distance += self.edges[edge].distance
        final_edge = (tour[len(tour)-1], tour[0])
        total_distance += self.edges[final_edge].distance
        return total_distance

    def __create_nodes(self):
        for i in list(self.problem.get_nodes()):
            x, y = self.problem.node_coords[i]
            new_node = Node(i, x, y)
            self.nodes[i] = new_node

    def __create_edges(self):
        for edge in list(self.problem.get_edges()):
            if edge[0] == edge[1]:
                continue
            origin = self.nodes[edge[0]]
            destiny = self.nodes[edge[1]]
            distance = self.problem.get_weight(*edge)
            new_edge = Edge(origin, destiny, distance)
            self.edges[edge] = new_edge


class Node:

    def __init__(self, index: int, x: float, y: float):
        self.index = index
        self.x = x
        self.y = y


class Edge:

    def __init__(self, origin: Node, destiny: Node, distance: float):
        self.origin = origin
        self.destiny = destiny
        self.distance = distance

    def print_distance(self):
        print(f'The distance from origin {self.origin.index} to destiny {self.destiny.index} is {self.distance}.')