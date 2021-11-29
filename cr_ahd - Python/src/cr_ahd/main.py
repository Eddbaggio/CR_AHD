import logging.config
import random
from pathlib import Path
from datetime import datetime
import cProfile

import utility_module.io as io
from solver_module import workflow as wf
from utility_module import utils as ut, evaluation as ev, cr_ahd_logging as log

logging.config.dictConfig(log.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    def cr_ahd():
        # setup
        logger.info(f'START {datetime.now()}')
        random.seed(0)

        # select the files to be solved
        paths = sorted(list(io.input_dir.iterdir()), key=ut.natural_sort_key)
        run, rad, n = 8, 1, 1  # rad: 0->150; 1->200; 2->300 // n: 0->10; 1->15
        # i = run * 6 + rad * 2 + n
        i = random.choice(range(len(paths)))
        paths = paths[:36]

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
                color=('solution_algorithm', 'num_int_auctions', 'fin_auction_num_auction_rounds'),
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
        secondary_parameter = 'neighborhoods'
        ev.print_top_level_stats(df, [secondary_parameter])

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
