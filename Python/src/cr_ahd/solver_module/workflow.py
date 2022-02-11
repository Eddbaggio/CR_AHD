import itertools
import logging
import logging.config
import multiprocessing
from datetime import datetime
from pprint import pformat

from tqdm import tqdm

from core_module import instance as it, solution as slt
from solver_module import solver as slv
from utility_module import profiling as pr, cr_ahd_logging as log, io

logging.config.dictConfig(log.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def execute_jobs(paths, configs, num_threads: int = 1, fail_on_error: bool = False):
    print(f'Solving on {num_threads} thread(s)')
    if num_threads > 1:
        console_log_level = logging.ERROR
    else:
        console_log_level = logging.INFO
    configs = list(configs)
    n_jobs = len(paths) * len(configs)
    jobs = itertools.product(paths, configs, [fail_on_error], [console_log_level])
    if num_threads == 1:
        solutions = [_execute_job(*j) for j in jobs]
    else:
        with multiprocessing.Pool(num_threads) as pool:
            solutions = list(tqdm(pool.imap(_execute_job_star, jobs), total=n_jobs))
    return solutions


def _execute_job_star(args):
    """ workaround to be able to display tqdm bar"""
    return _execute_job(*args)


def _execute_job(path, config, fail_on_error, console_log_level):
    log.remove_all_handlers(logging.getLogger())
    log_file_path = io.logging_dir.joinpath(f'{path.stem}_log.log')
    log.add_handlers(logging.getLogger(), str(log_file_path))

    logging.getLogger().handlers[0].setLevel(console_log_level)

    instance = it.read_vienna_instance(path)
    solver = slv.Solver(**config)
    try:
        timer = pr.Timer()
        tw_instance, solution = solver.execute(instance, None)
        timer.write_duration_to_solution(solution, 'runtime_total')
        # logger.info(f'{instance.id_}: Solved in {solution.timings["runtime_total"]}')
        solution.write_to_json()

    except Exception as e:
        logger.error(
            f'{e}\nFailed on instance {instance}\n'
            f'with solver\n'
            f'{pformat(solver.config, sort_dicts=False)}\n'
            f'at {datetime.now()}\n{e}')
        if fail_on_error:
            raise e
        else:
            solution = slt.CAHDSolution(instance)  # create an empty solution for failed instances
            solution.solver_config.update(solver.config)
            solution.write_to_json()

    return solution
