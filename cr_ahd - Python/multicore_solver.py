import multiprocessing
import os
from copy import deepcopy

import pandas as pd
from tqdm import tqdm

import instance as it
from solution_visitors.initializing_visitor import InitializingVisitor
from solution_visitors.local_search_visitor import FinalizingVisitor
from solution_visitors.routing_visitor import RoutingVisitor
from utils import path_input_custom, path_output_custom


# TODO: describe what this file is for and how it works exactly


def execute_all_routing_strategies(base_instance: it.Instance, centralized_flag: bool):
    """
    :param base_instance: (custom) instance that will we (deep)copied for each algorithm
    :param centralized_flag: TRUE if also a central (a single central carrier) instance (deep)copy shall be created
    and solved
    :return:
    """
    results = []

    # non-centralized instances
    for init_visitor in InitializingVisitor.__subclasses__():
        for routing_visitor in RoutingVisitor.__subclasses__():
            for local_search_visitor in FinalizingVisitor.__subclasses__():
                copy = deepcopy(base_instance)
                copy.initialize(init_visitor(2))
                copy.solve(routing_visitor(1))
                copy.finalize(local_search_visitor(2))
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
        results = execute_all_routing_strategies(base_instance,
                                                 centralized_flag, True,
                                                 False)  # TODO: extract the creation of the eval.csv file!!
        solomon_base_results.extend(results)
        centralized_flag = False

    performance = pd.DataFrame(solomon_base_results)
    performance = performance.set_index(['solomon_base', 'rand_copy', 'algorithm', 'num_carriers', 'num_vehicles'])

    # write the results
    write_dir = path_output_custom.joinpath(solomon)
    write_dir.mkdir(parents=True,
                    exist_ok=True)  # TODO All the directory creation should happen in the beginning somewhere once
    file_name = f'{base_instance.id_.split("#")[0]}eval.csv'
    write_path = write_dir.joinpath(file_name)
    performance.to_csv(write_path)


if __name__ == '__main__':
    # jobs = []
    # solomon_list = ['C101', 'C201', 'R101', 'R201', 'RC101', 'RC201']
    # for solomon in solomon_list:
    #     process = multiprocessing.Process(target=multi_func, args=(solomon, 3,))
    #     jobs.append(process)
    #     process.start()
    # for j in jobs:  # to ensure that program waits until all processes have finished before continuing
    #     j.join()

    # for running just a single instance
    multi_func('C101', num_of_inst=10)
