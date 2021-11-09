import datetime as dt
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from core_module import solution as slt

working_dir = Path().cwd()
print(working_dir.as_posix())
data_dir = working_dir.absolute().joinpath('data')
input_dir = data_dir.joinpath('Input')
assert input_dir.exists(), f'Input directory does not exist! Make sure the working directory (place from where ' \
                           f'execution is called) is correct '
output_dir = data_dir.joinpath('Output')
output_dir.mkdir(parents=True, exist_ok=True)


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


def solutions_to_df(solutions: List[slt.CAHDSolution], agg_level: str):
    """
    :param solutions: A List of solutions.
    :param agg_level: defines up to which level the solution will be summarized. E.g. if agg_level='carrier' the
    returned pd.DataFrame contains infos per carrier but not per tour since tours are summarized for each carrier.
    """

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
    df.drop(columns=['dist'], inplace=True)  # since the distance between depots is always 200 for the GH instances

    # set the multiindex
    # index = ['rad', 'n', 'run'] + list(solution.solver_config.keys())
    # if agg_level == 'carrier':
    #     index += ['carrier_id_']
    # if agg_level == 'tour':
    #     index += ['tour_id_']
    # df.set_index(keys=index, inplace=True)

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
