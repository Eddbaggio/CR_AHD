import logging.config
import multiprocessing
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

import utility_module.io
from core_module import instance as it, solution as slt
from solver_module import param_gen as pg, solver as slv
from utility_module import profiling as pr, cr_ahd_logging as log, utils as ut

logging.config.dictConfig(log.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def solve_with_parameters(instance: it.MDPDPTWInstance, parameter_generator=pg.parameter_generator):
    """
    :param parameter_generator:
    :param instance:
    :return: evaluation metrics (Instance.evaluation_metrics) of all the solutions obtained
    """
    solutions = []
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
                f'{e}\nFailed on instance {instance} with solver {solver.config} at {datetime.now()}')
            raise e
            solution = slt.CAHDSolution(instance)  # create an empty solution for failed instances
            solution.solver_config.update(solver.config)
            solution.write_to_json()
            solutions.append(solution)

    return solutions


def solve_single_instance(path: Path):
    """
    solves a single instance given by the path with all parameters defined in the parameter generator
    """
    log.remove_all_handlers(logging.getLogger())
    log_file_path = utility_module.io.output_dir.joinpath(f'{path.stem}_log.log')
    log.add_handlers(logging.getLogger(), str(log_file_path))

    instance = it.read_gansterer_hartl_mv(path)
    return solve_with_parameters(instance, pg.parameter_generator)


def solve_multiple_instances_multiprocessing(instance_paths):
    """
    solves multiple instances in parallel threads using the multiprocessing library
    """
    with multiprocessing.Pool(6) as pool:
        solutions = list(
            tqdm(pool.imap(solve_single_instance, instance_paths), total=len(instance_paths), desc="Parallel Solving",
                 disable=False))
    return solutions


def solve_multiple_instances(instance_paths):
    """
    solves multiple instances, given by their paths
    """
    solutions = []
    for path in tqdm(instance_paths, disable=True):
        solver_solutions = solve_single_instance(path)
        solutions.append(solver_solutions)
    return solutions


def _solve_instance_with_parameters(instance: it.MDPDPTWInstance, parameter_generator=pg.parameter_generator):
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


def solve_instances_multiprocessing(instance_paths):
    """
    solves multiple instances in parallel threads using the multiprocessing library
    """
    with multiprocessing.Pool(6) as pool:
        instance_solutions = list(tqdm(pool.imap(solve_instance, instance_paths),
                                       total=len(instance_paths),
                                       desc="Parallel Solving",
                                       disable=False))
    return [solution for instance_solutions in instance_solutions for solution in instance_solutions]


def solve_instances(instance_paths):
    """
    solves multiple instances, given by their paths
    """
    solutions = []
    for path in tqdm(instance_paths, disable=True):
        solver_solutions = solve_instance(path)
        solutions.extend(solver_solutions)
    return solutions
