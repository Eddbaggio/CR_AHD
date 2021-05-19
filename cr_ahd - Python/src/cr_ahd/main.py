import logging
import logging.config
import multiprocessing
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

import src.cr_ahd.solver as slv
import src.cr_ahd.utility_module.cr_ahd_logging as log
from src.cr_ahd.utility_module import utils as ut, plotting as pl
from src.cr_ahd.core_module import instance as it, solution as slt

# TODO write pseudo codes for ALL the stuff that's happening
# TODO: describe what THIS file is for and how it works exactly
# TODO local search: best improvement vs. first improvement!!
# TODO Better construction heuristic, better local search, metaheuristic

logging.config.dictConfig(log.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def execute_all(instance: it.PDPInstance):
    """
    :param instance: (custom) instance that will we (deep)copied for each algorithm
    :return: evaluation metrics (Instance.evaluation_metrics) of all the solutions obtained
    """
    solutions = []

    for solver in [
        # slv.Static,
        # slv.StaticCollaborative,
        slv.StaticCollaborativeAHD,
        # slv.Dynamic,
        # slv.DynamicCollaborative,
        # slv.DynamicAHD,
        # slv.DynamicCollaborativeAHD
    ]:
        solution = slt.GlobalSolution(instance)
        if instance.num_carriers == 1 and 'Auction' in solver.__name__:
            continue  # skip auction solvers for centralized instances
        logger.info(f'Solving {instance.id_} via {solver.__name__}')
        solver().execute(instance, solution)
        pl.plot_solution_2(instance, solution,
                           title=f'{instance.id_} with Solver "{solver.__name__} - Total profit: {solution.sum_profit()}"',
                           show=True)
        solution.write_to_json()
        solutions.append(solution)
    return solutions


def read_and_execute_all(path: Path):
    log.remove_all_file_handlers(logging.getLogger())
    log_file_path = ut.path_output_gansterer.joinpath(f'{path.stem}_log.log')
    log.add_file_handler(logging.getLogger(), str(log_file_path))

    instance = it.read_gansterer_hartl_mv(path)
    solutions = execute_all(instance)
    return solutions


def read_and_execute_all_parallel(paths):
    with multiprocessing.Pool() as pool:
        solutions = list(tqdm(pool.imap(read_and_execute_all, paths), total=len(paths)))
    df = write_solutions_to_multiindex_df(solutions)
    return df


def write_solutions_to_multiindex_df(solutions_per_instance: List[List[slt.GlobalSolution]]):
    """

    :param solutions_per_instance: A List of Lists. Each sublist contains solutions for a specific instance, each of
    them obtained from a different solution algorithm
    :return:
    """
    df = []
    for instance_solutions in solutions_per_instance:
        for solution in instance_solutions:
            for carrier in range(solution.num_carriers()):
                for tour in range(solution.carriers[carrier].num_tours()):
                    d = solution.carriers[carrier].tours[tour].summary()
                    d['id_'] = solution.id_
                    d.update(solution.meta)
                    d['num_carriers'] = solution.num_carriers()
                    d['solution_algorithm'] = solution.solution_algorithm
                    d['carrier_id_'] = carrier
                    d['tour_id_'] = tour
                    df.append(d)
    df = pd.DataFrame.from_records(df)
    df.set_index(
        keys=['id_', 'run', 'dist', 'rad', 'n', 'num_carriers', 'solution_algorithm', 'carrier_id_', 'tour_id_'],
        inplace=True)
    for column in df.select_dtypes(include=['timedelta64']):
        df[column] = df[column].dt.total_seconds()
    df.to_csv(ut.unique_path(ut.path_output_gansterer, 'evaluation' + '_#{:03d}' + '.csv'))
    df.to_excel(ut.unique_path(ut.path_output_gansterer, 'evaluation' + '_#{:03d}' + '.xlsx'), merge_cells=False)
    return df


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

    # paths = [Path('../../../data/Input/Gansterer_Hartl/3carriers/MV_instances/test.dat')]
    paths = list(Path('../../../data/Input/Gansterer_Hartl/3carriers/MV_instances/').iterdir())[4:5]
    for path in paths:
        read_and_execute_all(path)
    # df = read_and_execute_all_parallel(paths)
    logger.info('END')
