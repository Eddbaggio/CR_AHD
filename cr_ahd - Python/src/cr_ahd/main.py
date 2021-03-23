import multiprocessing
from copy import deepcopy
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

import src.cr_ahd.solving_module.solver as sl
from src.cr_ahd.core_module import instance as it
from src.cr_ahd.utility_module.evaluation import bar_plot_with_errors, plotly_bar_plot
from src.cr_ahd.utility_module.utils import path_input_custom, path_output_custom


# TODO write pseudo codes for ALL the stuff that's happening

# TODO how do carriers 'buy' requests from others? Is there some kind of money exchange happening?

# TODO re-integrate (animated) plots for
#  (1) static/dynamic sequential cheapest insertion construction
#  (2) dynamic construction
#  (3) I1 insertion construction

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

    for solver in [
        # sl.StaticSequentialInsertion,
        # sl.StaticI1Insertion,
        # sl.StaticI1InsertionWithAuction,
        # sl.DynamicSequentialInsertion,
        sl.DynamicI1Insertion,
        sl.DynamicI1InsertionWithAuctionA,
        sl.DynamicI1InsertionWithAuctionB
    ]:
        # print(f'Solving {base_instance.id_} with {solver.__name__}...')
        copy = deepcopy(instance)
        if instance.num_carriers == 1 and 'Auction' in solver.__name__:
            continue  # skip auction solvers for centralized instances
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

    with multiprocessing.Pool() as pool:
        results_collection = list(
            tqdm(pool.imap(read_and_execute_all, custom_files_paths), total=len(custom_files_paths)))
    flat_results_collection = [r for results in results_collection for r in results]
    grouped = write_results(flat_results_collection)  # rename function; not clear what are the results?
    return grouped


def write_results(result_collection: list):
    """
    write the collected outputs of the execute_all function to csv files. One file is created for each solomon_base

    :param result_collection:  (flat) list of single results of the execute_all function
    :return: the solomon_base grouped dataframe of results
    """
    performance = pd.DataFrame(result_collection)
    performance = performance.set_index(
        ['solomon_base', 'rand_copy', 'solution_algorithm', 'num_carriers', 'num_vehicles'])
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
    # read_and_execute_all(Path("../../../data/Input/Custom/C101/C101_3_15_ass_#002.json"))  # single collaborative
    # read_and_execute_all(Path("../../../data/Input/Custom/C201/C201_1_45_ass_#001.json"))  # single centralized
    grouped_evaluations = read_and_execute_all_parallel(n=5, which=solomon_list)  # multiple

    plotly_bar_plot(solomon_list, attributes=['num_act_veh', 'cost', ])
    # bar_plot_with_errors(solomon_list, attributes=['num_act_veh', 'cost', ])
