import multiprocessing
from copy import deepcopy
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import instance as it
from solving.Solver import Solver
from helper.utils import path_input_custom, path_output_custom


# TODO write pseudo codes for ALL the stuff that's happening

# TODO how do carriers 'buy' requests from others? Is there some kind of money exchange happening?

# TODO storing the vehicle assignment with each vertex (also in the file) may greatly simplify a few things.
#  Alternatively, store the assignment in the instance? Or have some kind of AssignmentManager class?! Another
#  possibility would be to have a class hierarchy for nodes: base_node, tw_node, depot_node, customer_node,
#  assigned_node, ... <- using inheritance and polymorphism

# TODO which of the @properties should be converted to proper class attributes, i.e. without delaying their
#  computation? the @property may slow down the code, BUT in many cases it's probably a more idiot-proof way
#  because otherwise I'd have to update the attribute which can easily be forgotten

# TODO re-integrate animated plots for
#  (1) static/dynamic sequential cheapest insertion construction => DONE
#  (2) dynamic construction
#  (3) I1 insertion construction => DONE

# TODO's with * are from 06/12/20 or later from when I tried to understand my own code

# TODO create class hierarchy! E.g. vertex (base, tw_vertex, depot_vertex, assigned_vertex, ...) and instance(
#  base, centralized_instance, ...)

# TODO: describe what THIS file is for and how it works exactly


def execute_all_visitors(base_instance: it.Instance, centralized_flag: bool = True):
    """
    :param base_instance: (custom) instance that will we (deep)copied for each algorithm
    :param centralized_flag: TRUE if also a central (a single central carrier) instance (deep)copy shall be created
    and solved
    :return: evaluation metrics (Instance.evaluation_metrics) of all the solutions obtained
    """
    results = []

    # non-centralized instances

    for solver in Solver.__subclasses__():
        print(f'Solving {base_instance.id_} with {solver.__name__}...')
        copy = deepcopy(base_instance)
        solver().solve(copy)
        copy.write_solution_to_json()
        results.append(copy.evaluation_metrics)

    # for r in results:
    #     for k, v in r.items():
    #         print(f'{k}:\t {v}')
    #
    # print()

    """algorithms_and_parameters = [
        (it.Instance.static_I1_construction, dict()),
        (it.Instance.dynamic_construction, dict(with_auction=False)),
        (it.Instance.dynamic_construction, dict(with_auction=True))
    ]
    for algorithm, parameters in algorithms_and_parameters:
        if not two_opt_only_flag:
            instance_copy = deepcopy(base_instance)
            algorithm(instance_copy, **parameters)
            instance_copy.write_solution_to_json()
            results.append(instance_copy.evaluation_metrics)
        # ====== 2 opt
        if two_opt_flag:
            instance_copy.two_opt()
            instance_copy.write_solution_to_json()
            results.append(instance_copy.evaluation_metrics)"""

    # centralized instances
    """if centralized_flag:
        algorithms_and_parameters = [
            (it.Instance.static_CI_construction, dict()),
            (it.Instance.static_I1_construction, dict(init_method='earliest_due_date'))
        ]
        # create a centralized instance, i.e. only one carrier
        centralized_instance = base_instance.to_centralized(base_instance.carriers[0].depot.coords)
        for algorithm, parameters in algorithms_and_parameters:
            if not two_opt_only_flag:
                instance_copy = deepcopy(centralized_instance)
                algorithm(instance_copy, **parameters)
                instance_copy.write_solution_to_json()
                results.append(instance_copy.evaluation_metrics)
            # ====== 2 opt
            if two_opt_flag:
                instance_copy.two_opt()
                instance_copy.write_solution_to_json()
                results.append(instance_copy.evaluation_metrics)"""

    return results


def read_and_execute_all_visitors(path: Path):
    """reads CUSTOM instance from a given path before executing all available visitors"""
    instance = it.read_custom_json_instance(path)
    results = execute_all_visitors(instance)
    return results


def read_n_and_execute_all_visitors_parallel(n=100):
    """reads the first n variants of the CUSTOM instances and runs all visitors via multiprocessing"""
    custom_directories = path_input_custom.iterdir()
    custom_files_paths = []
    for cd in custom_directories:
        custom_files_paths.extend(list(cd.iterdir())[:n])

    result_collection = []
    with multiprocessing.Pool() as pool:
        for results in tqdm(pool.map(read_and_execute_all_visitors, custom_files_paths), total=len(custom_files_paths)):
            result_collection.extend(results)

    performance = pd.DataFrame(result_collection)
    performance = performance.set_index(['solomon_base', 'rand_copy', 'initializing_visitor', 'routing_visitor',
                                         'finalizing_visitor', 'num_carriers', 'num_vehicles'])
    grouped = performance.groupby('solomon_base')
    # write the results
    for name, group in grouped:
        write_dir = path_output_custom.joinpath(name)
        write_dir.mkdir(parents=True,
                        exist_ok=True)  # TODO All the directory creation should happen in the beginning somewhere once
        file_name = f'{name}_custom_eval.csv'
        write_path = write_dir.joinpath(file_name)
        group.to_csv(write_path)

    return grouped


'''
def multi_func(solomon, num_of_inst):
    """
    splits different instances to multiple threads/cores
    for the first num_of_inst custom instances that exists in the solomon type folder, this function runs all available
    algorithms to solve each of these instances. The results are collected, transformed into a pd.DataFrame and saved as
    .csv
    :param solomon: solomon base class to be used. will read custom made classes from this base class
    :param num_of_inst: number of instances to read and solve
    """
    name = multiprocessing.current_process().name
    pid = os.getpid()
    print(f'{solomon} in {name} - {pid}')

    directory = path_input_custom.joinpath(solomon)
    instance_paths = list(directory.iterdir())[:num_of_inst]  # find the first num_of_inst custom instances
    centralized_flag = True  # TRUE if also the centralized versions should be solved
    solomon_base_results = []  # collecting all the results
    for inst_path in tqdm(iterable=instance_paths):
        path = directory.joinpath(inst_path)
        base_instance = it.read_custom_json_instance(path)
        results = execute_all_visitors(base_instance, centralized_flag)
        solomon_base_results.extend(results)
        centralized_flag = False

    performance = pd.DataFrame(solomon_base_results)
    performance = performance.set_index(['solomon_base', 'rand_copy', 'initializing_visitor', 'routing_visitor',
                                         'finalizing_visitor', 'num_carriers', 'num_vehicles'])

    # write the results
    write_dir = path_output_custom.joinpath(solomon)
    write_dir.mkdir(parents=True,
                    exist_ok=True)  # TODO All the directory creation should happen in the beginning somewhere once
    file_name = f'{base_instance.id_.split("#")[0]}eval.csv'
    write_path = write_dir.joinpath(file_name)
    performance.to_csv(write_path)
'''

if __name__ == '__main__':
    # jobs = []
    # solomon_list = ['C101', 'C201', 'R101', 'R201', 'RC101', 'RC201']
    # for solomon in solomon_list:
    #     process = multiprocessing.Process(target=multi_func, args=(solomon, 1,))
    #     jobs.append(process)
    #     process.start()
    # for j in jobs:  # to ensure that program waits until all processes have finished before continuing
    #     j.join()

    # NEW
    read_and_execute_all_visitors(Path(
        "C:/Users/Elting/ucloud/PhD/02_Research/02_Collaborative Routing for Attended Home Deliveries/01_Code/data/Input/Custom/C101/C101_3_15_ass_#001.json"))
    # grouped_evaluations = read_n_and_execute_all_visitors_parallel(3)
