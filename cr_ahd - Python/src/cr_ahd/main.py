import logging.config
import random
import re
import sys
from pathlib import Path
import os
from datetime import datetime
import cProfile

import utility_module.io as io
from solver_module import workflow as wf
from utility_module import utils as ut, evaluation as ev, cr_ahd_logging as log

logging.config.dictConfig(log.LOGGING_CONFIG)
logger = logging.getLogger(__name__)
logger.parent.handlers[0].setFormatter(log.CustomFormatter())


def instance_selector(run=None, rad=None, n=None):
    """

    :param run:
    :param rad:
    :param n:
    :return:
    """

    if not any([run, rad, n]):
        paths = sorted(list(io.input_dir.glob('*.dat')), key=ut.natural_sort_key)
        i = random.choice(range(len(paths)))
        return paths[i:i + 1]

    else:
        if isinstance(run, int):
            p_run = run
        elif isinstance(run, (list, tuple, range)):
            p_run = f"({'|'.join((str(x) for x in run))})"
        elif run == '*':
            p_run = '\d+'
        else:
            raise ValueError

        if isinstance(rad, int):
            p_rad = rad
        elif isinstance(rad, (list, tuple, range)):
            p_rad = f"({'|'.join((str(x) for x in rad))})"
        elif rad == '*':
            p_rad = '\d+'
        else:
            raise ValueError

        if isinstance(n, int):
            p_n = n
        elif isinstance(n, (list, tuple, range)):
            p_n = f"({'|'.join((str(x) for x in n))})"
        elif n == '*':
            p_n = '\d+'
        else:
            raise ValueError

        pattern = re.compile(f'run={p_run}\+dist=200\+rad={p_rad}\+n={p_n}(\.dat)')
        paths = []
        for file in (sorted(io.input_dir.glob('*.dat'), key=ut.natural_sort_key)):
            if pattern.match(file.name):
                paths.append(file)
        return paths


if __name__ == '__main__':
    def cr_ahd():
        # setup
        sys.argv[1]  # TODO parse command line arguments
        logger.info(f'START {datetime.now()}')
        # random.seed(0)

        paths = instance_selector(run=range(2), rad='*', n='*')
        # paths = instance_selector()

        # solving
        if len(paths) < 6:
            solutions = wf.solve_instances(paths)
        else:
            solutions = wf.solve_instances_multiprocessing(paths)

        agg_level = 'solution'
        df = io.solutions_to_df(solutions, agg_level)

        # write df
        io.output_dir.mkdir(exist_ok=True, parents=True)
        csv_path = io.unique_path(io.output_dir, 'evaluation_agg_' + agg_level + '_#{:03d}' + '.csv')
        df.to_csv(path_or_buf=csv_path, index=False)

        # plotting and evaluation
        ev.plot(df,
                values='sum_profit',
                color=('solution_algorithm', 'request_acceptance_attractiveness', 'max_num_accepted_infeasible',),
                category=('run',),
                facet_col=('rad',),
                facet_row=('n',),
                title=str(csv_path.name),
                html_path=io.unique_path(io.output_dir, 'CAHD_#{:03d}.html').as_posix(),
                )

        # ev.bar_chart(df,
        #              title=str(io.unique_path(io.output_dir, 'evaluation_agg_' + agg_level + '_#{:03d}' + '.csv').name),
        #              values='sum_profit',
        #              color=['solution_algorithm', 'num_int_auctions'],
        #              category='run',
        #              facet_col='rad',
        #              facet_row='n',
        #              show=True,
        #              html_path=io.unique_path(io.output_dir, 'CAHD_#{:03d}.html').as_posix())
        secondary_parameter = ['request_acceptance_attractiveness', 'max_num_accepted_infeasible']
        ev.print_top_level_stats(df, secondary_parameter)

        logger.info(f'END {datetime.now()}')

        # send windows to sleep
        # os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")


    # PROFILING
    cProfile.run('cr_ahd()', io.output_dir.joinpath('cr_ahd_stats'))
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
