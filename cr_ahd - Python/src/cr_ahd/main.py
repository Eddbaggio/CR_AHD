import logging
import logging.config
import multiprocessing
import pstats
import random
from copy import deepcopy
from pathlib import Path
from typing import List
from datetime import datetime
import cProfile
import pandas as pd
from tqdm import tqdm

import src.cr_ahd.solver_module.solver as slv
from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.solver_module.param_gen import parameter_generator
from src.cr_ahd.utility_module import utils as ut, evaluation as ev, cr_ahd_logging as log, profiling as pr

logging.config.dictConfig(log.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def solve_with_all_solvers(instance: it.MDPDPTWInstance, plot=False):
    """
    :param instance:
    :return: evaluation metrics (Instance.evaluation_metrics) of all the solutions obtained
    """
    solutions = []
    isolated_planning_starting_solution = None
    for solver_params in parameter_generator():
        solver = slv.Solver(**solver_params)
        try:
            timer = pr.Timer()

            # fixme: using a starting solution is tricky when intermediate auctions are possible
            """
            if not solver.acceptance.auction:  # isolated planning as starting point for collaborative
                tw_instance, solution = solver.execute(instance, None)
                starting_solution = solution
            else:  # collaborative planning can use starting solution & instance having the assigned time windows
                tw_instance, solution = solver.execute(tw_instance, starting_solution)
            """

            tw_instance, solution = solver.execute(instance, None)

            timer.write_duration_to_solution(solution, 'runtime_total')
            # logger.info(f'{instance.id_}: Solved in {solution.timings["runtime_total"]}')
            solution.write_to_json()
            solutions.append(deepcopy(solution))

        except Exception as e:
            logger.error(
                f'{e}\nFailed on instance {instance} with solver {solver.__class__.__name__} at {datetime.now()}')
            # raise e
            solution = slt.CAHDSolution(instance)  # create an empty solution for failed instances
            solver.update_solution_solver_config(solution)
            solution.write_to_json()
            solutions.append(solution)

    return solutions


def s_solve(path: Path, plot=False):
    """
    solves a single instance given by the path
    """
    log.remove_all_file_handlers(logging.getLogger())
    log_file_path = ut.output_dir_GH.joinpath(f'{path.stem}_log.log')
    log.add_file_handler(logging.getLogger(), str(log_file_path))

    instance = it.read_gansterer_hartl_mv(path)
    return solve_with_all_solvers(instance, plot)


def m_solve_multi_thread(instance_paths):
    """
    solves multiple instances in parallel threads using the multiprocessing library
    """
    with multiprocessing.Pool(6) as pool:
        solutions = list(
            tqdm(pool.imap(s_solve, instance_paths), total=len(instance_paths), desc="Parallel Solving", disable=False))
    return solutions


def m_solve_single_thread(instance_paths, plot=False):
    """
    solves multiple instances, given by their paths
    """
    solutions = []
    for path in tqdm(instance_paths, disable=True):
        solver_solutions = s_solve(path, plot=plot)
        solutions.append(solver_solutions)
    return solutions


def write_solution_summary_to_multiindex_df(solutions_per_instance: List[List[slt.CAHDSolution]], agg_level='tour'):
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
    # df.fillna('None', inplace=True)

    # set the multiindex

    index = ['rad', 'n', 'run'] + ut.solver_config
    if agg_level == 'carrier':
        index += ['carrier_id_']
    if agg_level == 'tour':
        index += ['tour_id_']
    df.set_index(keys=index, inplace=True)

    # convert timedelta to seconds
    for column in df.select_dtypes(include=['timedelta64']):
        df[column] = df[column].dt.total_seconds()

    # write to disk
    df.to_csv(ut.unique_path(ut.output_dir_GH, 'evaluation_agg_' + agg_level + '_#{:03d}' + '.csv'))
    df.to_excel(ut.unique_path(ut.output_dir_GH, 'evaluation_agg_' + agg_level + '_#{:03d}' + '.xlsx'),
                merge_cells=False)
    return df.reset_index().fillna('None').set_index(keys=index)


if __name__ == '__main__':
    def cr_ahd():
        logger.info(f'START {datetime.now()}')
        random.seed()

        paths = sorted(
            list(Path('../../../data/Input/Gansterer_Hartl/3carriers/MV_instances/').iterdir()),
            key=ut.natural_sort_key
        )

        run, rad, n = 1, 1, 1  # rad: 0->150; 1->200; 2->300 // n: 0->10; 1->15
        i = random.choice(range(len(paths)))
        # i = run * 6 + rad * 2 + n
        paths = paths[:]

        if len(paths) < 6:
            solutions = m_solve_single_thread(paths, plot=True)
        else:
            solutions = m_solve_multi_thread(paths)

        df = write_solution_summary_to_multiindex_df(solutions, 'solution')
        secondary_parameter = 'tour_improvement'
        ev.bar_chart(df,
                     title='',
                     values='sum_profit',
                     color=['solution_algorithm', secondary_parameter, ],
                     # color=secondary_parameter,
                     # category='rad', facet_col=None, facet_row='n',
                     category='run', facet_col='rad', facet_row='n',
                     # category='solution_algorithm', facet_col=None, facet_row=None,
                     show=True,
                     html_path=ut.unique_path(ut.output_dir_GH, 'CAHD_#{:03d}.html').as_posix())

        ev.print_top_level_stats(df, [secondary_parameter])

        logger.info(f'END {datetime.now()}')

        # send windows to sleep
        # os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")


    # PROFILING
    cProfile.run('cr_ahd()', ut.output_dir.joinpath('cr_ahd_stats'))
    """
    # STATS
    p = pstats.Stats(ut.output_dir.joinpath('cr_ahd_stats').as_posix())
    # remove the extraneous path from all the module names:
    p.strip_dirs()
    # sorts the profile by cumulative time in a function, and then only prints the n most significant lines:
    p.sort_stats('cumtime').print_stats(50)
    # see what functions were looping a lot, and taking a lot of time:
    p.sort_stats('tottime').print_stats(20)
    p.sort_stats('ncalls').print_stats(20)
    # p.print_callers(20)
    """
