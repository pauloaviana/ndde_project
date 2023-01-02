from population import Individual, NovelTrevisanIndividual
from typing import List
import numpy as np


def best_two(best: Individual, r1: Individual, r2: Individual, r3: Individual,
             r4: Individual, mutation_rate_f: float) -> np.array:
    vector_best = best.real_gene
    vector_r1 = r1.real_gene
    vector_r2 = r2.real_gene
    vector_r3 = r3.real_gene
    vector_r4 = r4.real_gene
    mutant_vector = vector_best + (mutation_rate_f * (vector_r1 - vector_r2)) + (
                mutation_rate_f * (vector_r3 - vector_r4))
    return mutant_vector


def trigonometric_mutation(r1: Individual, r2: Individual, r3: Individual) -> np.array:

    p = np.abs(r1.fitness) + np.abs(r2.fitness) + np.abs(r3.fitness)
    p1 = np.abs(r1.fitness) / p
    p2 = np.abs(r2.fitness) / p
    p3 = np.abs(r3.fitness) / p

    vector_r1 = r1.real_gene
    vector_r2 = r2.real_gene
    vector_r3 = r3.real_gene

    mutant_vector = (vector_r1 + vector_r2 + vector_r3) / 3
    mutant_vector += ((p2 - p1) * (vector_r1 - vector_r2))
    mutant_vector += ((p3 - p2) * (vector_r2 - vector_r3))
    mutant_vector += ((p1 - p3) * (vector_r1 - vector_r3))

    return mutant_vector


def de_rand_one(population: List[Individual], mutation_rate_f: float):

    for target in population:
        rng = np.random.default_rng()
        indexes = rng.choice(len(population), size=3, replace=False)
        r1 = population[indexes[0]]
        r2 = population[indexes[1]]
        r3 = population[indexes[2]]

        vector_r1 = r1.real_gene
        vector_r2 = r2.real_gene
        vector_r3 = r3.real_gene
        target.mutant_gene = vector_r1 + mutation_rate_f * (vector_r2 - vector_r3)


def de_rand_one_trevisan(population: List[NovelTrevisanIndividual], mutation_rate_f: float):

    for target in population:
        rng = np.random.default_rng()
        indexes = rng.choice(len(population), size=3, replace=False)
        r1 = population[indexes[0]]
        r2 = population[indexes[1]]
        r3 = population[indexes[2]]

        vector_r1 = r1.vector_gene
        vector_r2 = r2.vector_gene
        vector_r3 = r3.vector_gene
        target.mutant_gene = vector_r1 + mutation_rate_f * abs(vector_r2-vector_r3)


def de_best_two(population: List[Individual], mutation_rate_f: float):

    best = max(population, key=lambda individual: individual.fitness)
    for target in population:
        rng = np.random.default_rng()
        indexes = rng.choice(len(population), size=4, replace=False)

        r1 = population[indexes[0]]
        r2 = population[indexes[1]]
        r3 = population[indexes[2]]
        r4 = population[indexes[3]]

        vector_best = best.real_gene
        vector_r1 = r1.real_gene
        vector_r2 = r2.real_gene
        vector_r3 = r3.real_gene
        vector_r4 = r4.real_gene
        mutant_vector = vector_best + (mutation_rate_f * (vector_r1 - vector_r2))
        mutant_vector += mutation_rate_f * (vector_r3 - vector_r4)
        target.mutant_gene = mutant_vector


def de_best_two_trevisan(population: List[NovelTrevisanIndividual], best_individual: NovelTrevisanIndividual,
                         mutation_rate_f: float):

    for target in population:
        rng = np.random.default_rng()
        indexes = rng.choice(len(population), size=4, replace=False)

        r1 = population[indexes[0]]
        r2 = population[indexes[1]]
        r3 = population[indexes[2]]
        r4 = population[indexes[3]]

        vector_best = best_individual.vector_gene
        vector_r1 = r1.vector_gene
        vector_r2 = r2.vector_gene
        vector_r3 = r3.vector_gene
        vector_r4 = r4.vector_gene
        mutant_vector = vector_best + (mutation_rate_f * (vector_r1 - vector_r2))
        mutant_vector += mutation_rate_f * (vector_r3 - vector_r4)
        target.mutant_gene = abs(mutant_vector / max(mutant_vector))


def de_trigonometric_mutation(population: List[Individual]):

    for target in population:
        rng = np.random.default_rng()
        indexes = rng.choice(len(population), size=3, replace=False)
        r1 = population[indexes[0]]
        r2 = population[indexes[1]]
        r3 = population[indexes[2]]

        p = np.abs(r1.fitness) + np.abs(r2.fitness) + np.abs(r3.fitness)
        p1 = np.abs(r1.fitness) / p
        p2 = np.abs(r2.fitness) / p
        p3 = np.abs(r3.fitness) / p

        vector_r1 = r1.real_gene
        vector_r2 = r2.real_gene
        vector_r3 = r3.real_gene

        mutant_vector = (vector_r1 + vector_r2 + vector_r3) / 3
        mutant_vector += ((p2 - p1) * (vector_r1 - vector_r2))
        mutant_vector += ((p3 - p2) * (vector_r2 - vector_r3))
        mutant_vector += ((p1 - p3) * (vector_r1 - vector_r3))

        target.mutant_gene = mutant_vector