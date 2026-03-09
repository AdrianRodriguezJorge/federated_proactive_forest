"""
Módulo de utilidades para operaciones comunes en árboles de decisión y ensembles.
"""
import numpy as np
import random


def all_instances_same_class(x):
    return len(np.unique(x)) == 1


def categorical_data(x):
    return isinstance(x[0], str)


def bin_count(x, length):
    results = np.zeros(length, dtype=int)
    for i in x:
        results[i] += 1
    return results.tolist()


def count_classes(x):
    return len(np.unique(x))


def check_positive_array(x):
    array = np.array(x)
    return all(array > 0)


def check_array_sum_one(x):
    array = np.array(x)
    return sum(array) == 1


def get_instances(features_id, sample_size, probabilities):
    selected = None
    if (len(features_id) == len(probabilities) and
            sample_size <= len(features_id) and
            len(features_id) > 0 and sample_size > 0):
        selected = [0] * sample_size
        distribution = [0.0] * len(probabilities)
        total = 0
        for i in range(len(probabilities)):
            total += probabilities[i]
            distribution[i] = total
        for i in range(len(selected)):
            rd = random.random() * total
            for j in range(len(distribution)):
                if rd <= distribution[j]:
                    selected[i] = features_id[j]
                    break
    return selected
