import numpy as np
import pandas as pd

from numpy import sin, cos, exp, pi
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from typing import List

from individual import Jordan_Individual


'''

 All funtions to be used in jordan_de or j_mode goes here.

'''
# graph modelling

def polar_to_cartesian(r,theta):
    x = r * cos(theta)
    y = r * sin(theta)
    return x,y

def circle_distribution(path, filename):
    filepath = path+"/"+filename
    graph_dataframe = pd.read_csv(filepath, header=0, names=['vertice_A', 'vertice_B', 'weight'])

    total_vertices = 0
    for vertice in graph_dataframe.vertice_B:
        if vertice > total_vertices:
            total_vertices = vertice

    graph_matrix = np.zeros((total_vertices,total_vertices))
    #make incidence matrix here
    

    radian_spacing = 2* pi / total_vertices
    graph_coordinates = []
    for i in range(60):
        node_coordinates = polar_to_cartesian(1,radian_spacing*i)
        graph_coordinates.append(node_coordinates)
    
    return graph_dataframe, graph_coordinates



# Fitness evaluation functions

def maxcut_evaluation(graph_matrix,partition_list): #Partition Evaluation for MaxCut problem
    edges_count = 0
    for i in range(len(partition_list)-1):
        for j in range(len(partition_list)-i-1):
            if partition_list[i] != partition_list[i+j+1]:
                edges_count = edges_count + graph_matrix[i][i+j+1]
                
    return edges_count

def minbisec_evaluation(graph_matrix, partition_list): #Partition Evaluation for Min-Bisection problem
    edges_count = 0
    for i in range(len(partition_list)-1):
        for j in range(len(partition_list)-i-1):
            if partition_list[i] != partition_list[i+j+1]:
                edges_count = edges_count + graph_matrix[i][i+j+1]
    
    nodes_ratio = partition_list.count(0)/partition_list.count(1)
    return edges_count,nodes_ratio

# Jordan partition

def jordan_partition(graph_coordinates,polygon):
    # turn graph_coordinates into a list of shapely.Points:

    # turn polygon into a shapely.Polygon

    #Do partition here:
    partition_list = [0] * len(graph_coordinates)
    for i in range(len(graph_coordinates)):
        x = polygon.contains(graph_coordinates[i])
        print(x)
        if x == True:
            partition_list[i] = 1
    return partition_list


def rand_one(population: List[Jordan_Individual],mutation_rate_F: float):
    for target in population:
        rng = np.random.default_rng()
        indexes = rng.choice(len(population), size=3, replace=False)
        r1 = population[indexes[0]]
        r2 = population[indexes[1]]
        r3 = population[indexes[2]]

    vector_r1 = r1.polygon_gene
    vector_r2 = r2.polygon_gene
    vector_r3 = r3.polygon_gene
    target.mutant_gene = vector_r1 + mutation_rate_F * (vector_r2 - vector_r3)


def exponential_crossover(population: List[Jordan_Individual],crossover_rate: float):
    for target in population:
        target_len = len(target.polygon_gene)
        target.trial_polygon = target.polygon_gene
        start = np.random.uniform(0,target_len)
        target.trial_polygon[start] = target.mutant_gene
        temp_mutant = target.mutant_gene[start:target_len] + target.mutant_gene[:start]
        temp_trial = target.trial_polygon[start:target_len] + target.trial_polygon[:start]
        for i in range(target_len-1):
            r = np.random()
            if r < crossover_rate and i < target_len:
                temp_trial[i+1] = temp_mutant[i+1]
            else:
                break
        target.trial_polygon = temp_trial[target_len-2:] + temp_trial[:target_len-2]

def pairwise_selection(population: List[Jordan_Individual],graph_matrix,graph_coordinates):
    for target in population:
        trial_partition = jordan_partition(graph_coordinates,target.trial_polygon)
        trial_fitness = maxcut_evaluation(graph_matrix,trial_partition)
        if trial_fitness >= target.fitness:
            target.polygon_gene = target.trial_polygon
            target.fitness = trial_fitness
            target.associated_partition = trial_partition
    

            


g_matrix = [[0,1,0,0,0,1],[1,0,1,0,1,0],[0,1,0,1,0,0],[0,0,1,0,1,1],[0,1,0,1,0,0],[1,0,0,1,0,0]]
partition_1 = [0,0,0,1,1,1]

#print(minbisec_evaluation(g_matrix,partition_1))
#print(maxcut_evaluation(g_matrix,partition_1))




#TEST DATA

'''
polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
graph = [Point(0.5,0.5),Point(-1,0.5),Point(0.5,2),Point(0.3,0.2),Point(0.7,-3)]

print(jordan_partition(graph,polygon))
'''