import logging.config
import random
from pathlib import Path
from typing import List
from datetime import datetime
import cProfile
import pandas as pd

from src.cr_ahd.core_module import solution as slt
from src.cr_ahd.solver_module import workflow as wf
from src.cr_ahd.utility_module import utils as ut, evaluation as ev, cr_ahd_logging as log

logging.config.dictConfig(log.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


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
    csv_path = ut.unique_path(ut.output_dir_GH, 'evaluation_agg_' + agg_level + '_#{:03d}' + '.csv')
    df.to_csv(path_or_buf=csv_path)
    df.to_excel(ut.unique_path(ut.output_dir_GH, 'evaluation_agg_' + agg_level + '_#{:03d}' + '.xlsx'),
                merge_cells=False)
    return df.reset_index().fillna('None').set_index(keys=index), csv_path


if __name__ == '__main__':
    def cr_ahd():
        # setup
        logger.info(f'START {datetime.now()}')
        random.seed()

        # select the files to be solved
        paths = sorted(
            list(Path('../../../data/Input/Gansterer_Hartl/3carriers/MV_instances/').iterdir()),
            key=ut.natural_sort_key
        )
        run, rad, n = 11, 0, 1  # rad: 0->150; 1->200; 2->300 // n: 0->10; 1->15
        i = run * 6 + rad * 2 + n
        i = random.choice(range(len(paths)))
        paths = paths[:]

        # solving
        if len(paths) < 6:
            solutions = wf.solve_instances(paths, plot=True)
        else:
            solutions = wf.solve_instances_multiprocessing(paths)
        df, csv_path = write_solution_summary_to_multiindex_df(solutions, 'solution')

        # plotting and evaluation
        ev.bar_chart(df,
                     title=str(csv_path.name),
                     values='sum_travel_distance',
                     color=['tour_improvement'],
                     category='tour_improvement_time_limit_per_carrier',
                     facet_col='rad',
                     facet_row='n',
                     show=True,
                     html_path=ut.unique_path(ut.output_dir_GH, 'CAHD_#{:03d}.html').as_posix())
        secondary_parameter = 'neighborhoods'
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
