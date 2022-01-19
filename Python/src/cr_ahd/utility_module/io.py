import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utility_module import utils as ut

working_dir = Path().cwd()
print(working_dir.as_posix())
data_dir = working_dir.absolute().joinpath('data')
input_dir = data_dir.joinpath('Input')
assert input_dir.exists(), f'Input directory does not exist! Make sure the working directory (place from where ' \
                           f'execution is called) is correct '
output_dir = data_dir.joinpath('Output')
output_dir.mkdir(parents=True, exist_ok=True)
logging_dir = output_dir.joinpath('logs')
logging_dir.mkdir(parents=True, exist_ok=True)
solution_dir = output_dir.joinpath('solutions')
solution_dir.mkdir(parents=True, exist_ok=True)


class MyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, dt.datetime):
            return obj.isoformat()
        if isinstance(obj, dt.timedelta):
            return obj.total_seconds()
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)


# class CAHDSolutionSummaryCollection:
#     def __init__(self, solutions: List[Dict]):
#         self.summaries = solutions


def solutions_to_df(solutions, agg_level: str):
    """
    :param solutions: A List of solutions.
    :param agg_level: defines up to which level the solution will be summarized. E.g. if agg_level='carrier' the
    returned pd.DataFrame contains infos per carrier but not per tour since tours are summarized for each carrier.
    """
    assert solutions, f'No solutions available'
    df = []
    for solution in solutions:
        if agg_level == 'solution':
            record = solution.summary()
            record.pop('carrier_summaries')
            df.append(record)

        elif agg_level == 'carrier':
            raise NotImplementedError('override of dictionary is not secure yet. E.g. timings will be copied')
            for carrier in solution.carriers:
                record = solution.summary()
                record.pop('carrier_summaries')
                record.update(carrier.summary())
                record.pop('tour_summaries')
                df.append(record)

        elif agg_level == 'tour':
            raise NotImplementedError('override of dictionary is not secure yet. E.g. timings will be copied')
            for carrier in solution.carriers:
                for tour in carrier.tours:
                    record = solution.summary()
                    record.pop('carrier_summaries')
                    record.update(carrier.summary())
                    record.pop('tour_summaries')
                    record.update(tour.summary())
                    df.append(record)

        else:
            raise ValueError('agg_level must be one of "solution", "carrier" or "tour"')

    df = pd.DataFrame.from_records(df)

    # convert timedelta to seconds
    for column in df.select_dtypes(include=['timedelta64']):
        df[column] = df[column].dt.total_seconds()

    return df


def unique_path(directory, name_pattern) -> Path:
    """
    construct a unique numbered file name based on a template.
    Example template: file_name + '_#{:03d}' + '.json'

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
        permission = input(f'Should files and directories that exist at {path} be overwritten?\t[y/n]: ')
        if permission == 'y':
            return True
        else:
            raise FileExistsError
    else:
        return True


def vrptw_instance_selector(distance=None, num_carriers=None, num_requests=None, service_area_overlap=None, run=None):
    """

    :param distance: distance of the depots from the city center
    :param num_carriers: number of carriers
    :param num_requests: number of requests per carrier
    :param service_area_overlap: degree of overlap of the service areas between 0 (no overlap) and 1 (all carriers serve the whole city)
    :param run: run, i.e. which of the random instances with the above parameters
    :return: a MDVRPTWInstance
    """
    # DISTANCE
    if isinstance(distance, int):
        p_distance = distance
    elif distance is None:
        p_distance = '\d+'
    elif isinstance(distance, (list, tuple, range)):
        p_distance = f"({'|'.join((str(x) for x in distance))})"
    else:
        raise ValueError(f'distance must be int or list of int. distance={distance} is type {type(distance)}')

    # NUM_CARRIERS
    if isinstance(num_carriers, int):
        p_num_carriers = num_carriers
    elif num_carriers is None:
        p_num_carriers = '\d+'
    elif isinstance(num_carriers, (list, tuple, range)):
        p_num_carriers = f"({'|'.join((str(x) for x in num_carriers))})"
    else:
        raise ValueError(
            f'num_carriers must be int or list of int. num_carriers={num_carriers} is type {type(num_carriers)}')

    # NUM_REQUESTS
    if isinstance(num_requests, int):
        p_num_requests = num_requests
    elif num_requests is None:
        p_num_requests = '\d+'
    elif isinstance(num_requests, (list, tuple, range)):
        p_num_requests = f"({'|'.join((str(x) for x in num_requests))})"
    else:
        raise ValueError(
            f'num_requests must be int or list of int. num_requests={num_requests} is type {type(num_requests)}')

    # SERVICE_AREA_OVERLAP
    if isinstance(service_area_overlap, float):
        p_service_area_overlap = f'{int(service_area_overlap * 100):03d}'
    elif service_area_overlap is None:
        p_service_area_overlap = '\d+'
    elif isinstance(service_area_overlap, (list, tuple, range)):
        p_service_area_overlap = f"({'|'.join((f'{int(x * 100):03d}' for x in service_area_overlap))})"
    else:
        raise ValueError(
            f'service_area_overlap must be float or list of float. service_area_overlap={service_area_overlap} is type {type(service_area_overlap)}')

    # RUN
    if isinstance(run, int):
        p_run = f'{run:02d}'
    elif run is None:
        p_run = '\d+'
    elif isinstance(run, (list, tuple, range)):
        p_run = f"({'|'.join((f'{x:02d}' for x in run))})"
    else:
        raise ValueError(f'run must be int or list of int. run={run} is type {type(run)}')

    pattern = re.compile(f't=vienna\+d={p_distance}\+c={p_num_carriers}\+n={p_num_requests}\+'
                         f'o={p_service_area_overlap}\+r={p_run}(\.json)')  # run={p_run}\+dist=200\+rad={p_rad}\+n={p_n}(\.dat)')
    paths = []
    for file in (sorted(input_dir.glob('*.json'), key=ut.natural_sort_key)):
        if pattern.match(file.name):
            paths.append(file)
            # print(file.name)
    if len(paths) == 0:
        raise ValueError
    return paths


def instance_selector(run=None, rad=None, n=None):
    """
    If no arguments are passed a single, random Gansterer&Hartl instance is being solved.

    :param run:
    :param rad:
    :param n:
    :return:
    """
    # print(f'instance selector: run={run}({type(run)}), rad={rad}({type(rad)}), n={n}({type(n)})')
    if isinstance(run, int):
        p_run = run
    elif run is None:
        p_run = '\d+'
    elif isinstance(run, (list, tuple, range)):
        p_run = f"({'|'.join((str(x) for x in run))})"
    else:
        raise ValueError(f'run must be int or list of int. run={run} is type {type(run)}')

    if isinstance(rad, int):
        p_rad = rad
    elif rad is None:
        p_rad = '\d+'
    elif isinstance(rad, (list, tuple, range)):
        p_rad = f"({'|'.join((str(x) for x in rad))})"
    else:
        raise ValueError(f'rad must be int or list of int. rad={rad} is type {type(rad)}')

    if isinstance(n, int):
        p_n = n
    elif n is None:
        p_n = '\d+'
    elif isinstance(n, (list, tuple, range)):
        p_n = f"({'|'.join((str(x) for x in n))})"
    else:
        raise ValueError(f'n must be int or list of int. n={n} is type {type(run)}')

    pattern = re.compile(f'run={p_run}\+dist=200\+rad={p_rad}\+n={p_n}(\.dat)')
    paths = []
    for file in (sorted(input_dir.glob('*.dat'), key=ut.natural_sort_key)):
        if pattern.match(file.name):
            paths.append(file)
            # print(file.name)
    if len(paths) == 0:
        raise ValueError
    return paths
