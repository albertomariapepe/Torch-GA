"""Operations for constructing the Cayley 3-tensor needed
for the geometric product. Used internally.
"""
from itertools import combinations

import torch

from torchga.blades import get_normal_ordered  # Make sure to adjust this import based on your file structure


def _collapse_same(x):
    """Removes duplicate adjacent elements in a sorted list and returns the updated list."""
    for i in range(len(x) - 1):
        a, b = x[i], x[i + 1]
        if a == b:
            return False, x[:i] + x[i + 2:], a
    return True, x, None


def _reduce_bases(a, b, metric):
    """Reduces two bases based on the metric provided, keeping track of the sign."""
    if a == "":
        return 1, b
    elif b == "":
        return 1, a

    combined = list(a + b)

    # Bring into normal order
    sign, combined = get_normal_ordered(combined)

    done = False
    while not done:
        done, combined, combined_elem = _collapse_same(combined)
        if not done:
            sign *= metric[combined_elem]

    return sign, "".join(combined)


def blades_from_bases(vector_bases):
    """Generates all possible blade combinations from vector bases."""
    all_combinations = [""]
    degrees = [0]
    for i in range(1, len(vector_bases) + 1):
        combs = combinations(vector_bases, i)
        combs = ["".join(c) for c in combs]
        all_combinations += combs
        degrees += [i] * len(combs)
    return all_combinations, degrees


def get_cayley_tensor(metric, bases, blades):
    """Constructs the Cayley 3-tensor using the provided metric, bases, and blades."""


    num_blades = len(blades)

    t_geom = torch.zeros((num_blades, num_blades, num_blades), dtype=torch.int32)
    t_inner = torch.zeros((num_blades, num_blades, num_blades), dtype=torch.int32)
    t_outer = torch.zeros((num_blades, num_blades, num_blades), dtype=torch.int32)

    metric_dict = {bases[i]: metric[i] for i in range(len(bases))}

    for a in blades:
        for b in blades:
            sign, result = _reduce_bases(a, b, metric_dict)
            a_index = blades.index(a)
            b_index = blades.index(b)
            out_index = blades.index(result)
            t_geom[a_index, b_index, out_index] = sign

            # Degree went down -> part of inner
            if len(result) == abs(len(a) - len(b)):
                t_inner[a_index, b_index, out_index] = sign

            # Degree went up -> part of outer
            if len(result) == len(a) + len(b):
                t_outer[a_index, b_index, out_index] = sign

    return t_geom.to(torch.float32), t_inner.to(torch.float32), t_outer.to(torch.float32)
