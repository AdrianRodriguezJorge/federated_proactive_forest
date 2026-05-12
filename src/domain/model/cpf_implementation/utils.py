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
    # OPT-4: Use NumPy's C-optimized bincount instead of Python loop
    return np.bincount(x, minlength=length).tolist()


def count_classes(x):
    return len(np.unique(x))


def check_positive_array(x):
    array = np.array(x)
    return all(array > 0)


def check_array_sum_one(x):
    array = np.array(x)
    return sum(array) == 1


def get_instances(features_id, sample_size, probabilities):
    # OPT-5: Use NumPy's optimized weighted random sampling
    if (len(features_id) == len(probabilities) and
            sample_size <= len(features_id) and
            len(features_id) > 0 and sample_size > 0):
        probs = np.array(probabilities, dtype=np.float64)
        probs = probs / probs.sum()  # Normalize to ensure sum == 1.0
        return list(np.random.choice(features_id, size=sample_size, replace=True, p=probs))
    return None
