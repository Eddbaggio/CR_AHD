import datetime as dt
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


working_dir = Path().cwd()
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


def write_solution_summary_to_multiindex_df(solutions_per_instance: List[List], agg_level='tour'):
    """
    :param solutions_per_instance: A List of Lists of solutions. First Axis: instance, Second Axis: solver
    :param agg_level: defines up to which level the solution will be summarized. E.g. if agg_level='carrier' the
    returned pd.DataFrame contains infos per carrier but not per tour since tours are summarized for each carrier.
    """

    df = []
    for instance_solutions in solutions_per_instance:
        for solution in instance_solutions:

            if agg_level == 'solution':
                record = solution.meta.copy()  # rad, n, run, dist
                record.update(solution.solver_config)  # solution_algorithm, tour_construction, request_selection, ...
                record.update({k: v for k, v in solution.summary().items() if k != 'carrier_summaries'})
                df.append(record)

            elif agg_level == 'carrier':
                for carrier in range(solution.num_carriers()):
                    record = solution.meta.copy()  # rad, n, run, dist
                    record.update(solution.solver_config)
                    record.update(solution.carriers[carrier].summary())
                    record['carrier_id_'] = carrier
                    record.pop('tour_summaries')
                    df.append(record)

            elif agg_level == 'tour':
                for carrier in range(solution.num_carriers()):
                    ahd_solution = solution.carriers[carrier]
                    for tour in range(len(ahd_solution.tours)):
                        record = solution.meta.copy()  # rad, n, run, dist
                        record.update(solution.solver_config)
                        record.update(solution.carriers[carrier].tours[tour].summary())
                        record['carrier_id_'] = carrier
                        record['tour_id_'] = tour
                        df.append(record)

            else:
                raise ValueError

    df = pd.DataFrame.from_records(df)
    df.drop(columns=['dist'], inplace=True)  # since the distance between depots is always 200 for the GH instances

    # set the multiindex
    index = ['rad', 'n', 'run'] + solver_config
    if agg_level == 'carrier':
        index += ['carrier_id_']
    if agg_level == 'tour':
        index += ['tour_id_']
    df.set_index(keys=index, inplace=True)

    # convert timedelta to seconds
    for column in df.select_dtypes(include=['timedelta64']):
        df[column] = df[column].dt.total_seconds()

    # write to disk
    output_dir.mkdir(exist_ok=True, parents=True)
    csv_path = unique_path(output_dir, 'evaluation_agg_' + agg_level + '_#{:03d}' + '.csv')
    df.to_csv(path_or_buf=csv_path)
    # df.to_excel(unique_path(output_dir, 'evaluation_agg_' + agg_level + '_#{:03d}' + '.xlsx'),
    #             merge_cells=False)
    return df.reset_index().fillna('None').set_index(keys=index), csv_path


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
        permission = input(f'Should files and directories that exist at {path} be overwritten?\n[y/n]: ')
        if permission == 'y':
            return True
        else:
            raise FileExistsError
    else:
        return True