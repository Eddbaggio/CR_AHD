import functools
from pathlib import Path
import time
from collections import namedtuple
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import itertools

opts = {
    # 'num_trials': 10,
    'verbose': 3,
    'plot_level': 1,
    'speed_kmh': 60 ** 2,
    'start_time': 0,
    'alpha_1': 0.5,
    'mu': 1,
    'lambda': 2,
    'ccycler': plt.cycler(color=plt.get_cmap('Set1').colors)()
}

Coords = namedtuple('Coords', ['x', 'y'])
TimeWindow = namedtuple('TimeWindow', ['e', 'l'])
# Solomon_Instances = [file[:-4] for file in os.listdir('..\\data\\Input\\Solomon')]
path_project = Path.cwd().parent
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


def euclidean_distance(a: Coords, b: Coords):
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


if __name__ == '__main__':
    th = np.linspace(0, 2 * np.pi, 128)
    fig, ax = plt.subplots()
    ax.plot(th, np.cos(th), 'C1', label='C1')
    ax.plot(th, np.sin(th), 'C2', label='C2')
    ax.plot(th, np.sin(th + np.pi), 'C3', label='C3')
    ax.plot(th, np.cos(th + np.pi), 'C4', label='C4')
    ax.legend()
    plt.show()
