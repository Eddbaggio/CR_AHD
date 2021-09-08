import logging
import logging.config
import multiprocessing
import random
from copy import deepcopy
from pathlib import Path
from typing import List, Union, Tuple, Dict
from datetime import datetime
import os
import cProfile
import pstats
import pandas as pd
from tqdm import tqdm

import src.cr_ahd.solver as slv
from src.cr_ahd.auction_module import auction as au, request_selection as rs, bundle_generation as bg, bidding as bd, \
    winner_determination as wd, bundle_valuation as bv
from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import tour_construction as cns, metaheuristics as mh, neighborhoods as nh
from src.cr_ahd.tw_management_module import tw_management as twm, tw_selection as tws, tw_offering as two
from src.cr_ahd.utility_module import utils as ut, evaluation as ev, cr_ahd_logging as log, profiling as pr

logging.config.dictConfig(log.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def parameter_generator():
    """
    generate dicts with all parameters are required to initialize a slv.Solver.
    """
    neighborhoods: List[nh.Neighborhood] = [  # these are fixed at the moment, i.e. not looped over
        nh.PDPMove(),
        nh.PDPTwoOpt(),
        # nh.PDPRelocate()
    ]

    tour_constructions: List[cns.PDPParallelInsertionConstruction] = [
        cns.MinTravelDistanceInsertion(),
        # cns.MinTimeShiftInsertion()
    ]

    tour_improvements: List = [
        # mh.LocalSearchFirst([neighborhoods[0]]),
        # mh.LocalSearchFirst([neighborhoods[1]]),
        # mh.LocalSearchBest([neighborhoods[0]]),
        # mh.LocalSearchBest([neighborhoods[1]]),
        # mh.PDPTWSequentialLocalSearch(neighborhoods),
        # mh.PDPTWIteratedLocalSearch(neighborhoods),
        # mh.PDPTWVariableNeighborhoodDescent(neighborhoods),
        # mh.PDPTWReducedVariableNeighborhoodSearch(neighborhoods),
        # mh.PDPTWSimulatedAnnealing(neighborhoods),
        mh.NoMetaheuristic([]),
    ]

    time_window_managements: List[twm.TWManagement] = [
        twm.TWManagementSingle(two.FeasibleTW(),
                               tws.UnequalPreference()),
        # twm.TWManagementNoTW(None, None)
    ]

    nums_submitted_requests: List[int] = [
        # 3,
        4,
        # 5
    ]

    request_selections: List[rs.RequestSelectionBehavior.__class__] = [
        # rs.Random,
        # rs.SpatialBundleDSum,  # the original 'cluster' strategy by Gansterer & Hartl (2016)
        # rs.SpatialBundleDMax,
        # rs.MinDistanceToForeignDepotDMin,
        # rs.MarginalProfitProxy,
        # rs.MarginalProfitProxyNeighbor,
        # rs.ComboRaw,
        rs.ComboStandardized,
        # rs.LosSchulteBundle,
        # rs.TemporalRangeCluster,
        # TODO SpatioTemporalCluster is not yet good enough & sometimes even infeasible
        # rs.SpatioTemporalCluster,
    ]

    nums_auction_bundles: List[int] = [
        # 50,
        100,
        # 200,
        # 300,
        # 500
    ]

    bundle_generations: List[Tuple[bg.LimitedBundlePoolGenerationBehavior.__class__, Dict[str, float]]] = [
        (bg.GeneticAlgorithm, dict(population_size=300,
                                   num_generations=100,
                                   mutation_rate=0.5,
                                   generation_gap=0.9, )
         ),
        # (bg.BestOfAllBundlings, dict()),
        # (bg.RandomMaxKPartition, dict())
    ]

    bundling_valuations: List[bv.BundlingValuation.__class__] = [
        bv.GHProxyBundlingValuation,
        # bv.MinDistanceBundlingValuation,
        # bv.LosSchulteBundlingValuation,
        # bv.RandomBundlingValuation,
    ]

    for tour_construction in tour_constructions:
        for tour_improvement in tour_improvements:
            for time_window_management in time_window_managements:
                # Isolated Planning Parameters, no auction
                """
                yield dict(tour_construction=tour_construction,
                           tour_improvement=tour_improvement,
                           time_window_management=time_window_management,
                           auction=False,
                           )
               """
                for num_submitted_requests in nums_submitted_requests:
                    for request_selection in request_selections:
                        for num_auction_bundles in nums_auction_bundles:
                            for bundle_generation, bundle_generation_kwargs in bundle_generations:
                                for bundling_valuation in bundling_valuations:
                                    # auction for collaborative planning
                                    auction = au.Auction(tour_construction,
                                                         tour_improvement,
                                                         request_selection(num_submitted_requests),
                                                         bundle_generation(num_auction_bundles=num_auction_bundles,
                                                                           bundling_valuation=bundling_valuation(),
                                                                           **bundle_generation_kwargs
                                                                           ),
                                                         bidding=bd.DynamicInsertionAndImprove(tour_construction,
                                                                                               tour_improvement),
                                                         winner_determination=wd.MaxBidGurobiCAP1(),
                                                         )
                                    # collaborative planning
                                    yield dict(tour_construction=tour_construction,
                                               tour_improvement=tour_improvement,
                                               time_window_management=time_window_management,
                                               auction=auction,
                                               )
    pass


def execute_all(instance: it.MDPDPTWInstance, plot=False):
    """
    :param instance: (custom) instance that will we (deep)copied for each algorithm
    :return: evaluation metrics (Instance.evaluation_metrics) of all the solutions obtained
    """
    solutions = []
    isolated_planning_starting_solution = None
    for solver_params in parameter_generator():

        solver = slv.Solver(**solver_params)
        try:
            timer = pr.Timer()
            solution = solver.execute(instance,
                                      # isolated_planning_starting_solution
                                      )
            timer.write_duration_to_solution(solution, 'runtime_total')
            # if solution.solver_config['solution_algorithm'] == 'IsolatedPlanning':
            #     isolated_planning_starting_solution = solution
            solution.write_to_json()
            solutions.append(deepcopy(solution))

        except Exception as e:
            logger.error(f'{e}\nFailed on instance {instance} with solver {solver.__class__.__name__} at {datetime.now()}')
            raise e
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
            key=ut.natural_sort_key)
        run, rad, n = 12, 0, 1  # rad: 0->150; 1->200; 2->300 // n: 0->10; 1->15
        instance_idx = run * 6 + rad * 2 + n
        instance_idx = random.choice(range(len(paths)))
        paths = paths[instance_idx:instance_idx+1]

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
                     category='rad', facet_col=None, facet_row='n',
                     # category='run', facet_col='rad', facet_row='n',
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
    # sorts the profile by cumulative time in a function, and then only prints the ten most significant lines:
    p.sort_stats('cumtime').print_stats(50)
    # see what functions were looping a lot, and taking a lot of time:
    p.sort_stats('tottime').print_stats(20)
    p.sort_stats('ncalls').print_stats(20)
    # p.print_callers(20)
    """
