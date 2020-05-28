import pandas as pd
import numpy as np


def split_iterable(iterable, num_elements):
    k, m = divmod(len(iterable), num_elements)
    return (iterable[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_elements))


def euclidean_distance(a: tuple, b: tuple):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def make_dist_matrix(nodes: list):
    # assuming that nodes are of type rq.Request
    index = [i.id_ for i in nodes]
    dist_matrix = pd.DataFrame(index=index, columns=index)

    for i in nodes:
        for j in nodes:
            dist_matrix.loc[i.id_, j.id_] = euclidean_distance(i.coords, j.coords)
    return dist_matrix
