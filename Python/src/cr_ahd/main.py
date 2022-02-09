import logging.config
import os
from datetime import datetime

from solver_module import workflow as wf, config as pg
from utility_module import cr_ahd_logging as log, io, evaluation as ev
from utility_module.argparse_utils import parser

logging.config.dictConfig(log.LOGGING_CONFIG)
logger = logging.getLogger(__name__)
logger.parent.handlers[0].setFormatter(log.CustomFormatter())

if __name__ == '__main__':
    def cr_ahd():
        # if called from within PyCharm
        if "PYCHARM" in os.environ:
            args = {'distance': 7,
                    'num_carriers': 3,
                    'num_requests': [50, 100],
                    'carrier_max_num_tours': [3],
                    'service_area_overlap': [0, 0.5, 1.0],
                    'run': range(20),
                    'threads': 6,
                    'fail': 1,
                    }
        # else read from terminal parameters
        else:
            args = parser.parse_args().__dict__

        start = datetime.now()
        logger.info(f'START {start}')

        paths = io.vrptw_instance_selector(distance=args['distance'],
                                           num_carriers=args['num_carriers'],
                                           num_requests=args['num_requests'],
                                           carrier_max_num_tours=args['carrier_max_num_tours'],
                                           service_area_overlap=args['service_area_overlap'],
                                           run=args['run'],
                                           )
        configs = list(pg.configs())

        # solving
        solutions = wf.execute_jobs(paths, configs, args['threads'], args['fail'])

        agg_level = 'solution'
        df = io.solutions_to_df(solutions, agg_level)

        # write df
        csv_path = io.unique_path(io.output_dir, 'evaluation_agg_' + agg_level + '_#{:03d}' + '.csv')
        df.to_csv(path_or_buf=csv_path, index=False)

        collaboration_gains = ev.collaboration_gain(df)
        collaboration_gains.to_csv(str(csv_path).replace('agg_solution', 'coll_gain'), index=True)

        end = datetime.now()
        logger.info(f'END {end}')
        logger.info(f'DURATION {end - start}')


    cr_ahd()

    """
    # PROFILING
    cProfile.run('cr_ahd()', io.output_dir.joinpath('cr_ahd_stats'))

    # STATS
    p = pstats.Stats(io.output_dir.joinpath('cr_ahd_stats').as_posix())
    # remove the extraneous path from all the module names:
    p.strip_dirs()
    # sorts the profile by cumulative time in a function, and then only prints the n most significant lines:
    p.sort_stats('cumtime').print_stats(50)
    # see what functions were looping a lot, and taking a lot of time:
    p.sort_stats('tottime').print_stats(20)
    p.sort_stats('ncalls').print_stats(20)
    # p.print_callers(20)
    """
