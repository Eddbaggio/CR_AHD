import logging.config
import logging
import multiprocessing
import pickle
from copy import deepcopy
from pathlib import Path
import datetime as dt
from typing import List

import pandas as pd
from tqdm import tqdm

import src.cr_ahd.solving_module.solver as sl
import src.cr_ahd.utility_module.utils as ut
import src.cr_ahd.utility_module.cr_ahd_logging as log
from src.cr_ahd.core_module import instance as it

# TODO write pseudo codes for ALL the stuff that's happening

# TODO re-integrate (animated) plots for
#  (1) static/dynamic sequential cheapest insertion construction
#  (2) dynamic construction
#  (3) I1 insertion construction

# TODO: describe what THIS file is for and how it works exactly

# TODO use Memento Pattern for (e.g.) Tours and Vertices. This will avoid costly "undo" operations that are currently
#  handled specifically. E.g. atm an infeasible route section reversal (as in 2opt) will be reverted / "undone" by
#  re-reverting the attempt. maybe simply taking an old snapshot is less costly

# TODO two-opt: best improvement vs. first improvement!!
from src.cr_ahd.utility_module.evaluation import plotly_bar_plot

logging.config.dictConfig(log.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def execute_all(instance: it.Instance):
    """
    :param instance: (custom) instance that will we (deep)copied for each algorithm
    :return: evaluation metrics (Instance.evaluation_metrics) of all the solutions obtained
    """
    solution_summaries = []

    for solver in [
        # sl.StaticSequentialInsertion,
        # sl.StaticI1Insertion,
        # sl.StaticI1InsertionWithAuction,
        # sl.DynamicSequentialInsertion,
        sl.DynamicI1Insertion,
        # sl.DynamicI1InsertionWithAuctionA,
        # sl.DynamicI1InsertionWithAuctionB,
        sl.DynamicI1InsertionWithAuctionC,
    ]:
        # print(f'Solving {base_instance.id_} with {solver.__name__}...')
        copy = deepcopy(instance)
        if instance.num_carriers == 1 and 'Auction' in solver.__name__:
            continue  # skip auction solvers for centralized instances
        logger.info(f'Solving {instance.id_} via {solver.__name__}')
        solver().execute(copy)
        copy.write_solution_and_summary_to_json()
        solution_summaries.append(copy.solution_summary)
    return solution_summaries


def read_and_execute_all(path: Path):
    """reads CUSTOM instance from a given path before executing all available visitors"""
    log.remove_all_file_handlers(logging.getLogger())
    log_file_path = ut.path_output_custom.joinpath(path.name.split('_')[0], f'{path.name}_log.log' )
    log.add_file_handler(logging.getLogger(), str(log_file_path))
    instance = it.read_custom_json_instance(path)
    solution_summaries = execute_all(instance)
    return solution_summaries


def read_and_execute_all_parallel(n: int, which: List):
    """reads the first n variants of the CUSTOM instances specified by the solomon list and runs all visitors via
    multiprocessing

    :param n:number of randomized instances to consider
    :param which: which instances (defined by Solomon string) to consider
    """
    custom_files_paths = get_custom_instance_paths(n, which)

    with multiprocessing.Pool(8) as pool:
        solution_summaries_collection = list(
            tqdm(pool.imap(read_and_execute_all, custom_files_paths), total=len(custom_files_paths)))
    flat_solution_summaries_collection = [r for results in solution_summaries_collection for r in results]
    grouped = write_solution_summaries_df(flat_solution_summaries_collection)  # rename function; not clear what are the results?
    return grouped


def write_solution_summaries_df(solution_summaries: list):
    """
    write the collected outputs of the execute_all function to csv files. One file is created for each solomon_base

    :param solution_summaries: (flat) list of single results (i.e. the evaluation metrics) of the execute_all function
    :return: the solomon_base grouped dataframe of results
    """
    performance = pd.DataFrame(solution_summaries)
    for column in performance.select_dtypes(include=['timedelta64']):
        performance[column] = performance[column].dt.total_seconds()
    performance = performance.set_index(
        ['solomon_base', 'rand_copy', 'solution_algorithm', 'num_carriers', 'num_vehicles'])
    grouped = performance.groupby('solomon_base')

    for name, group in grouped:
        write_dir = ut.path_output_custom.joinpath(name)
        write_dir.mkdir(parents=True, exist_ok=True)
        file_name = f'{name}_custom_eval.csv'
        write_path = write_dir.joinpath(file_name)
        group.to_csv(write_path)
        group.to_excel(write_path.with_suffix('.xlsx'), merge_cells=False)

    return grouped


def get_custom_instance_paths(n, which: List):
    """retrieve the Path to the first n custom instances
    :param n: number of custom paths to retrieve, if None, then all available files are considered
    :param which: List of customized solomon name instances to consider"""

    custom_directories = (ut.path_input_custom.joinpath(s) for s in which)
    custom_files_paths = []
    for cd in custom_directories:
        custom_files_paths.extend(list(cd.iterdir())[:n])
    return custom_files_paths


if __name__ == '__main__':
    logger.info('START')
    solomon_list = ['C101', 'C201']  #, 'R101', 'R201', 'RC101', 'RC201']
    # solomon_list = ut.Solomon_Instances
    # read_and_execute_all(Path("../../../data/Input/Custom/C101/C101_3_15_ass_#002.json"))  # single collaborative
    # read_and_execute_all(Path("../../../data/Input/Custom/C201/C201_1_45_ass_#001.json"))  # single centralized
    grouped_evaluations = read_and_execute_all_parallel(n=2, which=solomon_list)  # multiple

    plotly_bar_plot(solomon_list, attributes=['num_act_veh', 'travel_distance'])  #, 'duration'
    # bar_plot_with_errors(solomon_list, attributes=['num_act_veh', 'cost', ])
    logger.info('END')
