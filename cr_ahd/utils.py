from collections import namedtuple

import numpy as np
import pandas as pd
from matplotlib.pyplot import cycler, get_cmap
from typing import List
import os


opts = {'num_trials': 10,
        'verbose': 0,
        'plot_level': 0,
        'speed_kmh': 60 ** 2,
        'start_time': 0,
        'alpha_1': 1,
        'mu': 1,
        'lambda': 0,
        'ccycler': cycler(color=get_cmap('Set1').colors)()
        }

Coords = namedtuple('Coords', ['x', 'y'])
TimeWindow = namedtuple('TimeWindow', ['e', 'l'])
Solomon_Instances = [file[:-4] for file in os.listdir('../data/Solomon')]


def split_iterable(iterable, num_chunks):
    """ splits an iterable, e.g. a list into num_chunks parts of roughly the same length. If no exact split is
    possible the first chunk(s) will be longer. """
    k, m = divmod(len(iterable), num_chunks)
    return (iterable[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_chunks))


def euclidean_distance(a: Coords, b: Coords):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def make_dist_matrix(vertices: List):
    """
    :param vertices: List of vertices each of class Vertex
    :return: pd.DataFrame distance matrix
    """
    # assuming that vertices are of type rq.Request
    index = [i.id_ for i in vertices]
    dist_matrix:pd.DataFrame = pd.DataFrame(index=index, columns=index, dtype='float64')

    for i in vertices:
        for j in vertices:
            dist_matrix.loc[i.id_, j.id_] = euclidean_distance(i.coords, j.coords)
    return dist_matrix


def travel_time(dist):
    return (dist / opts['speed_kmh']) * 60 ** 2  # compute time in seconds


class InsertionError(Exception):
    """Exception raised for errors in the insertion of a request into a tour.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
