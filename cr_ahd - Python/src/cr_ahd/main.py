import logging
import logging.config
import multiprocessing
from copy import deepcopy
from pathlib import Path
from typing import List
from datetime import datetime

import pandas as pd
from tqdm import tqdm

import src.cr_ahd.solver as slv
from src.cr_ahd.auction_module import auction as au, request_selection as rs, bundle_generation as bg, bidding as bd, \
    winner_determination as wd
from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import tour_construction as cns, metaheuristics as mh, local_search as ls
from src.cr_ahd.tw_management_module import tw_management as twm, tw_selection as tws, tw_offering as two
from src.cr_ahd.utility_module import utils as ut, evaluation as ev, cr_ahd_logging as log

logging.config.dictConfig(log.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def execute_all(instance: it.PDPInstance, plot=False):
    """
    :param instance: (custom) instance that will we (deep)copied for each algorithm
    :return: evaluation metrics (Instance.evaluation_metrics) of all the solutions obtained
    """
    solutions = []

    # define underlying modular elements and loop over all combinations
    neighborhoods = [ls.PDPMove(), ls.PDPTwoOpt()]
    for tour_construction in [
        cns.MinTravelDistanceInsertion(),
        # cns.MinTimeShiftInsertion()
    ]:
        for tour_improvement in [
            mh.PDPVariableNeighborhoodDescent(neighborhoods),
            # mh.NoMetaheuristic(neighborhoods)
        ]:
            for tw_management in [
                twm.TWManagementSingle(two.FeasibleTW(),
                                       tws.UnequalPreference()),
                # twm.TWManagementNoTW(None, None)
            ]:

                # Isolated Planning
                solver = slv.Solver(tour_construction, tour_improvement, tw_management, False)
                try:
                    solution = solver.execute(instance)
                    isolated_planning_starting_solution = solution
                    solution.write_to_json()
                    solutions.append(deepcopy(solution))

                except Exception as e:
                    # raise e
                    logger.error(f'{e}\nFailed on instance {instance} with solver {solver.__class__.__name__}')
                    solution = slt.CAHDSolution(instance)  # create an empty solution for failed instances
                    solver.update_solution_solver_config(solution)
                    solution.write_to_json()
                    solutions.append(solution)

                # Collaborative Planning
                for num_submitted_requests in [
                    4,
                    # 6
                ]:
                    for request_selection in [
                        rs.Random(num_submitted_requests),
                        rs.SpatialCluster(num_submitted_requests),
                        rs.TemporalRangeCluster(num_submitted_requests),
                        # rs.SpatioTemporalCluster(num_submitted_requests)  # TODO not yet good enough, somtimes infeasible
                    ]:
                        for num_auction_bundles in [
                            # 50,
                            100,
                            # 200,
                            # 300,
                            # 500
                        ]:

                            auction = au.Auction(tour_construction,
                                                 tour_improvement,
                                                 request_selection,
                                                 bg.GeneticAlgorithm(num_auction_bundles),
                                                 bd.DynamicReOptAndImprove(tour_construction, tour_improvement),
                                                 wd.MaxBidGurobiCAP1(),
                                                 )
                            solver = slv.Solver(tour_construction, tour_improvement, tw_management, auction)
                            try:
                                solution = solver.execute(instance, isolated_planning_starting_solution)
                                solution.write_to_json()
                                solutions.append(deepcopy(solution))

                            except Exception as e:
                                # raise e
                                logger.error(
                                    f'{e}\nFailed on instance {instance} with solver {solver.__class__.__name__}')
                                solution = slt.CAHDSolution(instance)  # create an empty solution for failed instances
                                solver.update_solution_solver_config(solution)
                                solution.write_to_json()
                                solutions.append(solution)

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

            if agg_level == 'solution':
                record = solution.meta.copy()  # rad, n, run, dist
                record.update(solution.solver_config)  # solution_algorithm, tour_construction, request_selection, ...
                record.update(solution.summary())
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
                    for tour in range(solution.carriers[carrier].num_tours()):
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

    index = ['rad', 'n', 'run'] + ev.solver_config
    if agg_level == 'carrier':
        index += ['carrier_id_']
    if agg_level == 'tour':
        index += ['tour_id_']
    df.set_index(keys=index, inplace=True)

    # convert timedelta to seconds
    for column in df.select_dtypes(include=['timedelta64']):
        df[column] = df[column].dt.total_seconds()

    # write to disk
    df.to_csv(ut.unique_path(ut.output_dir_GH, 'evaluation_' + agg_level + '_#{:03d}' + '.csv'))
    df.to_excel(ut.unique_path(ut.output_dir_GH, 'evaluation_' + agg_level + '_#{:03d}' + '.xlsx'), merge_cells=False)
    return df.reset_index().fillna('None').set_index(keys=index)


if __name__ == '__main__':
    logger.info(f'START {datetime.now()}')

    # paths = [Path('../../../data/Input/Gansterer_Hartl/3carriers/MV_instances/test.dat')]
    paths = sorted(
        list(Path('../../../data/Input/Gansterer_Hartl/3carriers/MV_instances/').iterdir()),
        key=ut.natural_sort_key)
    paths = paths[:48]

    # solutions = m_solve_single_thread(paths, plot=False)
    solutions = m_solve_multi_thread(paths)

    df = write_solution_summary_to_multiindex_df(solutions, 'carrier')
    ev.bar_chart(df,
                 title='4 requests selected',
                 values='sum_profit',
                 category='rad',
                 color=['solution_algorithm', 'request_selection'],
                 facet_col=None,
                 facet_row='n',
                 show=True,
                 html_path=ut.unique_path(ut.output_dir_GH, 'CAHD_#{:03d}.html').as_posix())

    ev.print_top_level_stats(df)

    logger.info(f'END {datetime.now()}')
