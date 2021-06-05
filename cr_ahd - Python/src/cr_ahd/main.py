import logging
import logging.config
import multiprocessing
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

import src.cr_ahd.solver as slv
import src.cr_ahd.utility_module.cr_ahd_logging as log
from src.cr_ahd.utility_module import utils as ut, plotting as pl, evaluation as ev
from src.cr_ahd.core_module import instance as it, solution as slt

# TODO write pseudo codes for ALL the stuff that's happening
# TODO: describe what THIS file is for and how it works exactly
# TODO local search: best improvement vs. first improvement!!
# TODO Better construction heuristic, better local search, metaheuristic

logging.config.dictConfig(log.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def execute_all(instance: it.PDPInstance, plot=False):
    """
    :param instance: (custom) instance that will we (deep)copied for each algorithm
    :return: evaluation metrics (Instance.evaluation_metrics) of all the solutions obtained
    """
    solutions = []

    for solver in [
        # slv.Static,
        # slv.StaticCollaborative,
        # slv.StaticCollaborativeAHD,

        # slv.Dynamic,
        # slv.DynamicCollaborative,
        # slv.DynamicCollaborativeSingleAuction,

        # slv.DynamicCollaborativeAHD,

        # slv.IsolatedPlanningNoTW,
        # slv.CollaborativePlanningNoTW,
        slv.IsolatedPlanning,
        slv.CollaborativePlanning,
    ]:
        solution = slt.CAHDSolution(instance)
        logger.info(f'{instance.id_}: Solving via {solver.__name__} ...')
        try:
            solver().execute(instance, solution)
            logger.info(f'{instance.id_}: Successfully solved via {solver.__name__}')
            if plot:
                pl.plot_solution_2(
                    instance,
                    solution,
                    title=f'{instance.id_} with Solver "{solver.__name__} - Total profit: {solution.sum_profit()}"',
                    show=True
                )
            solution.write_to_json()
            solutions.append(solution)

        except Exception as e:
            logger.error(f'{e}\tFailed on instance {instance} with solver {solver.__name__}')

    return solutions


def read_and_execute_all(path: Path, plot=False):
    log.remove_all_file_handlers(logging.getLogger())
    log_file_path = ut.output_dir_GH.joinpath(f'{path.stem}_log.log')
    log.add_file_handler(logging.getLogger(), str(log_file_path))

    instance = it.read_gansterer_hartl_mv(path)
    solutions = execute_all(instance, plot)
    return solutions


"""
def read_and_execute_all_parallel(paths):
    with multiprocessing.Pool() as pool:
        solutions = list(tqdm(pool.imap(read_and_execute_all, paths), total=len(paths), desc="Parallel Solving"))
    df = write_solutions_to_multiindex_df(solutions)
    return df
"""


def write_solutions_to_multiindex_df(solutions_per_instance: List[List[slt.CAHDSolution]]):
    """

    :param solutions_per_instance: A List of Lists of solutions. First Axis: instance, Second Axis: solver
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
    df.to_csv(ut.unique_path(ut.output_dir_GH, 'evaluation' + '_#{:03d}' + '.csv'))
    df.to_excel(ut.unique_path(ut.output_dir_GH, 'evaluation' + '_#{:03d}' + '.xlsx'), merge_cells=False)
    return df


if __name__ == '__main__':
    logger.info('START')

    # paths = [Path('../../../data/Input/Gansterer_Hartl/3carriers/MV_instances/test.dat')]
    paths = list(Path('../../../data/Input/Gansterer_Hartl/3carriers/MV_instances/').iterdir())[:6]
    solutions = []
    for path in paths:
        solver_solutions = read_and_execute_all(path, plot=False)
        solutions.append(solver_solutions)
    df = write_solutions_to_multiindex_df(solutions)
    ev.bar_chart(df)
    logger.info('END')
