import multiprocessing
from copy import deepcopy
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from solving_module.Solver import Solver
from src.cr_ahd.core_module import instance as it
from src.cr_ahd.solving_module.routing_visitor import RoutingVisitor
from src.cr_ahd.utility_module.evaluation import bar_plot_with_errors
from utility_module.utils import path_input_custom, path_output_custom


# TODO write pseudo codes for ALL the stuff that's happening

# TODO how do carriers 'buy' requests from others? Is there some kind of money exchange happening?

# TODO which of the @properties should be converted to proper class attributes, i.e. without delaying their
#  computation? the @property may slow down the code, BUT in many cases it's probably a more idiot-proof way
#  because otherwise I'd have to update the attribute which can easily be forgotten

# TODO re-integrate (animated) plots for
#  (1) static/dynamic sequential cheapest insertion construction
#  (2) dynamic construction
#  (3) I1 insertion construction

# TODO create class hierarchy! E.g. vertex (base, tw_vertex, depot_vertex, assigned_vertex, ...) and instance(
#  base, centralized_instance, ...)

# TODO: describe what THIS file is for and how it works exactly

# TODO use Memento Pattern for (e.g.) Tours and Vertices. This will avoid costly "undo" operations that are currently
#  handled specifically. E.g. atm an infeasible route section reversal (as in 2opt) will be reverted / "undone" by
#  re-reverting the attempt


def execute_all(instance: it.Instance):
    """
    :param instance: (custom) instance that will we (deep)copied for each algorithm
    :return: evaluation metrics (Instance.evaluation_metrics) of all the solutions obtained
    """
    results = []

    # non-centralized instances
    for solver in Solver.__subclasses__():
        # print(f'Solving {base_instance.id_} with {solver.__name__}...')
        copy = deepcopy(instance)
        solver().execute(copy)
        copy.write_solution_to_json()
        results.append(copy.evaluation_metrics)
    return results


def read_and_execute_all(path: Path):
    """reads CUSTOM instance from a given path before executing all available visitors"""
    instance = it.read_custom_json_instance(path)
    results = execute_all(instance)
    return results


def read_and_execute_all_parallel(n: int, which: List):
    """reads the first n variants of the CUSTOM instances specified by the solomon list and runs all visitors via
    multiprocessing

    :param n:number of randomized instances to consider
    :param which: which instances (defined by Solomon string) to consider
    """
    custom_files_paths = get_custom_instance_paths(n, which)

    result_collection = []
    with multiprocessing.Pool() as pool:
        for results in tqdm(pool.imap(read_and_execute_all, custom_files_paths),
                            total=len(custom_files_paths)):
            result_collection.extend(results)

    grouped = write_results(result_collection)  # rename function; not clear what are the results?
    return grouped


def write_results(result_collection: list):
    performance = pd.DataFrame(result_collection)
    performance = performance.set_index(['solomon_base', 'rand_copy', 'solution_algorithm', 'num_carriers', 'num_vehicles'])
    grouped = performance.groupby('solomon_base')
    # write the results
    for name, group in grouped:
        write_dir = path_output_custom.joinpath(name)
        write_dir.mkdir(parents=True,
                        exist_ok=True)  # TODO All the directory creation should happen in the beginning somewhere once
        file_name = f'{name}_custom_eval.csv'
        write_path = write_dir.joinpath(file_name)
        group.to_csv(write_path)
    return grouped


def get_custom_instance_paths(n, which: List):
    """retrieve the Path to the first n custom instances
    :param n: number of custom paths to retrieve, if None, then all available files are considered
    :param which: List of customized solomon name instances to consider"""

    custom_directories = (path_input_custom.joinpath(s) for s in which)
    custom_files_paths = []
    for cd in custom_directories:
        custom_files_paths.extend(list(cd.iterdir())[:n])
    return custom_files_paths


if __name__ == '__main__':
    solomon_list = ['C101', 'C201', 'R101', 'R201', 'RC101', 'RC201']
    # read_and_execute_all(Path("../../../data/Input/Custom/C101/C101_3_15_ass_#001.json"))  # single
    grouped_evaluations = read_and_execute_all_parallel(n=5, which=solomon_list)  # multiple
    bar_plot_with_errors(solomon_list,
                         columns=['num_act_veh',
                                  'cost',
                                  ],
                         )
