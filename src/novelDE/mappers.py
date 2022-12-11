import numpy as np
from scipy.stats import rankdata


def rank_order_value(tour: np.array) -> np.array:
    return rankdata(tour, method='min')


def backward_mapping(discrete_tour: np.array) -> np.array:
    real_tour = np.zeros(len(discrete_tour))

    random_tour = np.zeros(len(discrete_tour))
    random_tour[0] = np.random.random(1)[0]
    for i in range(1, len(real_tour)):
        random_tour[i] = random_tour[i-1] + np.random.random(1)[0]

    for i in range(len(discrete_tour)):
        node = discrete_tour[i]
        real_tour[i] = random_tour[node-1]

    return real_tour


