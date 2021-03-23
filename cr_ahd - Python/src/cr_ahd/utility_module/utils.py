import functools
import itertools
import random
import time
from collections import namedtuple
from pathlib import Path
from typing import List, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

opts = {
    # 'num_trials': 10,
    'verbose': 0,
    'plot_level': 1,
    'speed_kmh': 60 ** 2,
    'start_time': 0,
    'alpha_1': 0.5,
    'mu': 1,
    'lambda': 2,
    'ccycler': plt.cycler(color=plt.get_cmap('Set1').colors)(),
    'dynamic_cycle_time': 25,
}

Coordinates = namedtuple('Coords', ['x', 'y'])
TimeWindow = namedtuple('TimeWindow', ['e', 'l'])
# Solomon_Instances = [file[:-4] for file in os.listdir('..\\data\\Input\\Solomon')]
path_project = Path(
    'C:/Users/Elting/ucloud/PhD/02_Research/02_Collaborative Routing for Attended Home Deliveries/01_Code')
path_input = path_project.joinpath('data', 'Input')
path_input_custom = path_input.joinpath('Custom')
path_input_solomon = path_input.joinpath('Solomon')
Solomon_Instances = [file.stem for file in path_input_solomon.iterdir()]
path_output = path_project.joinpath('data', 'Output')
path_output_custom = path_output.joinpath('Custom')

# alpha 100%
univie_colors_100 = [
    '#0063A6',  # universitätsblau
    '#666666',  # universtitätsgrau
    '#A71C49',  # weinrot
    '#DD4814',  # orangerot
    '#F6A800',  # goldgelb
    '#94C154',  # hellgrün
    '#11897A',  # mintgrün
]
# alpha 80%
univie_colors_60 = [
    '#6899CA',  # universitätsblau
    '#B5B4B4',  # universtitätsgrau
    '#C26F76',  # weinrot
    '#F49C6A',  # orangerot
    '#FCCB78',  # goldgelb
    '#C3DC9F',  # hellgrün
    '#85B6AE',  # mintgrün
]
# paired
univie_colors_paired = list(itertools.chain(*zip(univie_colors_100, univie_colors_60)))

univie_cmap = LinearSegmentedColormap.from_list('univie', univie_colors_100, N=len(univie_colors_100))
univie_cmap_paired = LinearSegmentedColormap.from_list('univie_paired', univie_colors_paired,
                                                       N=len(univie_colors_100) + len(univie_colors_60))


def split_iterable(iterable, num_chunks):
    """ splits an iterable, e.g. a list into num_chunks parts of roughly the same length. If no exact split is
    possible the first chunk(s) will be longer. """
    k, m = divmod(len(iterable), num_chunks)
    return (iterable[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_chunks))


def euclidean_distance(a: Coordinates, b: Coordinates):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def make_dist_matrix(vertices: List):
    """
    :param vertices: List of vertices each of class Vertex
    :return: pd.DataFrame distance matrix
    """
    # assuming that vertices are of type rq.Request
    index = [i.id_ for i in vertices]
    dist_matrix: pd.DataFrame = pd.DataFrame(index=index, columns=index, dtype='float64')

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


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        if opts['verbose'] > 0:
            print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value, run_time

    return wrapper_timer


def unique_path(directory, name_pattern):
    """
    construct a unique numbered file name based on a template
    :param directory: directory which shall be the parent dir of the file
    :param name_pattern: pattern for the file name, with room for a counter
    :return: file path that is unique in the specified directory
    """
    counter = 0
    while True:
        counter += 1
        path = directory / name_pattern.format(counter)
        if not path.exists():
            return path


def ask_for_overwrite_permission(path: Path):
    if path.exists():
        permission = input(f'Should files and directories that exist at {path} be overwritten?\n[y/n]: ')
        if permission == 'y':
            return True
        else:
            raise FileExistsError
    else:
        return True


def get_carrier_by_id(carriers, id_):
    for c in carriers:
        if c.id_ == id_:
            return c
    raise ValueError


def powerset(iterable, include_empty_set=True):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    if include_empty_set:
        rng = range(len(s) + 1)
    else:
        rng = range(1, len(s) + 1)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in rng)


def flatten_dict_of_lists(d: dict):
    """d is Dict[Any, List[Any]] pairs. This unpacks it such that there is only a flat list of all values"""
    pool = []
    for _, v in d.items():
        pool.extend(v)
    return pool


def random_partition(li):
    min_chunk = 1
    max_chunk = len(li)
    it = iter(li)
    while True:
        randint = np.random.randint(min_chunk, max_chunk)
        nxt = list(itertools.islice(it, randint))
        if nxt:
            yield nxt
        else:
            break


def random_max_k_partition(ls, max_k) -> Dict[int, Tuple[Any, ...]]:
    """partition ls in at most k randomly sized disjoint subsets

    """
    # https://stackoverflow.com/a/45880095
    # we need to know the length of ls, so convert it into a list
    ls = list(ls)
    # sanity check
    if max_k < 1:
        return []
    # randomly determine the actual k
    k = random.randint(1, min(max_k, len(ls)))
    # Create a list of length ls, where each element is the index of
    # the subset that the corresponding member of ls will be assigned
    # to.
    #
    # We require that this list contains k different values, so we
    # start by adding each possible different value.
    indices = list(range(k))
    # now we add random values from range(k) to indices to fill it up
    # to the length of ls
    indices.extend([random.choice(list(range(k))) for _ in range(len(ls) - k)])
    # shuffle the indices into a random order
    random.shuffle(indices)
    # construct and return the random subset: sort the elements by
    # which subset they will be assigned to, and group them into sets
    partitions = dict()
    sortd = sorted(zip(indices, ls), key=lambda x: x[0])
    for index, subset in itertools.groupby(sortd, key=lambda x: x[0]):
        partitions[index] = tuple(
            x[1] for x in subset)  # TODO: better use frozenset (rather than tuple) for dict values?
    return partitions


def conjunction(*conditions):
    """
    combines multiple logical conditions such that all must hold

    :param conditions:
    :return:
    """
    return functools.reduce(np.logical_and, conditions)


def disjunction(*conditions):
    """
    combines multiple logical conditions such that at least one must hold

    :param conditions:
    :return:
    """
    return functools.reduce(np.logical_or, conditions)
