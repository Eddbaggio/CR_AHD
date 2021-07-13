import logging
import logging.config
import multiprocessing
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

import src.cr_ahd.solver as slv
import src.cr_ahd.utility_module.cr_ahd_logging as log
from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import tour_construction as cns, metaheuristics as mh
from src.cr_ahd.utility_module import utils as ut, plotting as pl, evaluation as ev

logging.config.dictConfig(log.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def execute_all(instance: it.PDPInstance, plot=False):
    """
    :param instance: (custom) instance that will we (deep)copied for each algorithm
    :return: evaluation metrics (Instance.evaluation_metrics) of all the solutions obtained
    """
    solutions = []

    construction_method = cns.MinTravelDistanceInsertion()
    improvement_method = mh.PDPVariableNeighborhoodDescent()
    # improvement_method = mh.NoMetaheuristic()

    for solver in [
        slv.IsolatedPlanningNoTW,
        slv.IsolatedPlanning,
        slv.CollaborativePlanningNoTW,
        slv.CollaborativePlanning,
        # slv.CentralizedPlanning,
    ]:
        logger.info(f'{instance.id_}: Solving via {solver.__name__} ...')
        fails = 0
        try:
            if solver.__name__ == 'IsolatedPlanningNoTW':
                solution = solver(construction_method, improvement_method).execute(instance)
                sol_iso_no_tw = solution
            elif solver.__name__ == 'CollaborativePlanningNoTW':
                # Collab: can take the isolated planning as a starting solution and will then only do the auction
                solution = solver(construction_method, improvement_method).execute(instance, sol_iso_no_tw)
            elif solver.__name__ == 'IsolatedPlanning':
                solution = solver(construction_method, improvement_method).execute(instance)
                sol_iso = solution
            elif solver.__name__ == 'CollaborativePlanning':
                # Collab: can take the isolated planning as a starting solution and will then only do the auction
                solution = solver(construction_method, improvement_method).execute(instance, sol_iso)

            logger.info(f'{instance.id_}: Successfully solved via {solver.__name__}')
            if plot:
                pl.plot_solution_2(instance, solution, show=True,
                                   title=f'{instance.id_} with Solver "{solver.__name__} - Total profit: {solution.sum_profit()}"')
            solution.write_to_json()
            solutions.append(solution)

        except Exception as e:
            # raise e
            logger.error(f'{e}\nFailed on instance {instance} with solver {solver.__name__}')
            solution = slt.CAHDSolution(instance)  # create an empty solution for failed instances?!
            solution.solution_algorithm = str(solver.__name__)
            solution.write_to_json()
            solutions.append(solution)
            fails += 1

    return solutions


def s_solve(path: Path, plot=False):
    log.remove_all_file_handlers(logging.getLogger())
    log_file_path = ut.output_dir_GH.joinpath(f'{path.stem}_log.log')
    log.add_file_handler(logging.getLogger(), str(log_file_path))

    instance = it.read_gansterer_hartl_mv(path)
    return execute_all(instance, plot)


def m_solve_multi_thread(instance_paths):
    with multiprocessing.Pool(6) as pool:
        solutions = list(
            tqdm(pool.imap(s_solve, instance_paths), total=len(instance_paths), desc="Parallel Solving", disable=False))
    return solutions


def m_solve_single_thread(instance_paths, plot=False):
    solutions = []
    for path in tqdm(instance_paths, disable=True):
        solver_solutions = s_solve(path, plot=plot)
        solutions.append(solver_solutions)
    return solutions


def write_solution_summary_to_multiindex_df(solutions_per_instance: List[List[slt.CAHDSolution]], agg_level='tour'):
    """
    :param solutions_per_instance: A List of Lists of solutions. First Axis: instance, Second Axis: solver
    """
    df = []
    for instance_solutions in solutions_per_instance:
        for solution in instance_solutions:
            for carrier in range(solution.num_carriers()):

                if agg_level == 'carrier':
                    d = solution.carriers[carrier].summary()
                    d['carrier_id_'] = carrier
                    d.pop('tour_summaries')
                    d.update(solution.meta)
                    d['solution_algorithm'] = solution.solution_algorithm
                    df.append(d)

                elif agg_level == 'tour':
                    for tour in range(solution.carriers[carrier].num_tours()):
                        d = solution.carriers[carrier].tours[tour].summary()
                        d.update(solution.meta)
                        d['solution_algorithm'] = solution.solution_algorithm
                        d['carrier_id_'] = carrier
                        d['tour_id_'] = tour
                        df.append(d)

    df = pd.DataFrame.from_records(df)
    df = df.drop(columns=['dist'])

    # set the multiindex
    index = ['rad', 'n', 'run', 'solution_algorithm', 'carrier_id_']
    if agg_level == 'tour':
        index += ['tour_id_']
    df.set_index(keys=index, inplace=True)

    # convert timedelta to seconds
    for column in df.select_dtypes(include=['timedelta64']):
        df[column] = df[column].dt.total_seconds()

    # write to disk
    df.to_csv(ut.unique_path(ut.output_dir_GH, 'evaluation_' + agg_level + '_#{:03d}' + '.csv'))
    df.to_excel(ut.unique_path(ut.output_dir_GH, 'evaluation_' + agg_level + '_#{:03d}' + '.xlsx'), merge_cells=False)
    return df


if __name__ == '__main__':
    logger.info('START')

    # paths = [Path('../../../data/Input/Gansterer_Hartl/3carriers/MV_instances/test.dat')]
    paths = sorted(
        list(Path('../../../data/Input/Gansterer_Hartl/3carriers/MV_instances/').iterdir()),
        key=ut.natural_sort_key)
    paths = paths[:24]

    # solutions = m_solve_single_thread(paths, plot=False)
    solutions = m_solve_multi_thread(paths)

    df = write_solution_summary_to_multiindex_df(solutions, 'carrier')
    ev.bar_chart(df,
                 title='VND(Move+2opt); 4 requests submitted;',
                 values='sum_profit',
                 category='run',
                 color='solution_algorithm',
                 facet_col='rad',
                 facet_row='n',
                 show=True,
                 html_path=ut.unique_path(ut.output_dir_GH, 'CAHD_#{:03d}.html').as_posix())

    ev.print_top_level_stats(df)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 0):

        grouped = df.groupby(['rad', 'n', 'run', 'solution_algorithm']).aggregate(
            {'num_tours': sum, 'sum_profit': sum, 'acceptance_rate': np.mean})

        # print(grouped[['num_tours', 'sum_profit', 'acceptance_rate']])

        for name, group in grouped.groupby('solution_algorithm'):
            print(f'{group["num_tours"].astype(bool).sum(axis=0)}/{len(group)} solved by {name}')
    logger.info('END')
