import itertools
import logging
import logging.config
import multiprocessing
from datetime import datetime

from tqdm import tqdm

import utility_module.io
from core_module import instance as it, solution as slt
from solver_module import solver as slv
from utility_module import profiling as pr, cr_ahd_logging as log

logging.config.dictConfig(log.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

'''
def _solve_instance_with_parameters(instance: it.MDVRPTWInstance, parameter_generator=pg.parameter_generator):
    """

    """
    solutions = []
    isolated_planning_starting_solution = None
    for solver_params in parameter_generator():
        solver = slv.Solver(**solver_params)
        try:
            timer = pr.Timer()
            tw_instance, solution = solver.execute(instance, None)
            timer.write_duration_to_solution(solution, 'runtime_total')
            # logger.info(f'{instance.id_}: Solved in {solution.timings["runtime_total"]}')
            solution.write_to_json()
            solutions.append(deepcopy(solution))

        except Exception as e:
            logger.error(
                f'{e}\nFailed on instance {instance} with solver {solver.__class__.__name__} at {datetime.now()}')
            raise e
            solution = slt.CAHDSolution(instance)  # create an empty solution for failed instances
            solution.solver_config.update(solver.config)
            solution.write_to_json()
            solutions.append(solution)

    return solutions


def solve_instance(path: Path):
    """
    solves a single instance given by the path
    """
    log.remove_all_handlers(logging.getLogger())
    log_file_path = utility_module.io.output_dir.joinpath(f'{path.stem}_log.log')
    log.add_handlers(logging.getLogger(), str(log_file_path))

    instance = it.read_gansterer_hartl_mv(path)
    return _solve_instance_with_parameters(instance)


def solve_instances(instance_paths, num_threads: int = 1):
    """
    solves instance files single- or multi-threaded
    """
    if num_threads == 1:
        solutions = []
        for path in tqdm(instance_paths, disable=True):
            solver_solutions = solve_instance(path)
            solutions.extend(solver_solutions)
        return solutions

    else:
        with multiprocessing.Pool(num_threads) as pool:
            instance_solutions = list(tqdm(pool.imap(solve_instance, instance_paths),
                                           total=len(instance_paths),
                                           desc="Parallel Solving",
                                           disable=False))
        return [solution for instance_solutions in instance_solutions for solution in instance_solutions]
'''


def execute_jobs(paths, configs, num_threads: int = 1, fail_on_error: bool = False):
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
    log_file_path = utility_module.io.output_dir.joinpath(f'{path.stem}_log.log')
    log.add_handlers(logging.getLogger(), str(log_file_path))

    logging.getLogger().handlers[0].setLevel(console_log_level)

    instance = it.read_gansterer_hartl_mv(path)
    solver = slv.Solver(**config)
    try:
        timer = pr.Timer()
        tw_instance, solution = solver.execute(instance, None)
        timer.write_duration_to_solution(solution, 'runtime_total')
        # logger.info(f'{instance.id_}: Solved in {solution.timings["runtime_total"]}')
        solution.write_to_json()

    except Exception as e:
        logger.error(
            f'{e}\nFailed on instance {instance} with solver {solver.config} at {datetime.now()}')
        if fail_on_error:
            raise e
        else:
            solution = slt.CAHDSolution(instance)  # create an empty solution for failed instances
            solution.solver_config.update(solver.config)
            solution.write_to_json()

    return solution
