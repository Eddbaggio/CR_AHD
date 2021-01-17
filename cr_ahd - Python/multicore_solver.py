import multiprocessing
import os
from pathlib import Path
from copy import deepcopy

import pandas as pd
from tqdm import tqdm

import instance as it
import evaluation as ev
from utils import path_input_custom, path_output_custom


# TODO: describe what this file is for and how it works exactly


def run_all_algorithms(base_instance: it.Instance, centralized_flag: bool):
    """
    :param base_instance: (custom) instance that will we (deep)copied for each algorithm
    :param centralized_flag: TRUE if also a central (a single central carrier) instance (deep)copy shall be created
    and solved
    :return:
    """
    results = []

    # non-centralized instances
    algorithms_and_parameters = [
        (it.Instance.static_I1_construction, dict()),
        (it.Instance.dynamic_construction, dict(with_auction=False)),
        (it.Instance.dynamic_construction, dict(with_auction=True))
    ]
    for algorithm, parameters in algorithms_and_parameters:
        instance_copy = deepcopy(base_instance)
        algorithm(instance_copy, **parameters)
        instance_copy.write_solution_to_json()
        results.append(instance_copy.evaluation_metrics)
        # ====== 2 opt
        instance_copy.two_opt()
        instance_copy.write_solution_to_json()
        results.append(instance_copy.evaluation_metrics)

    # centralized instances
    if centralized_flag:
        algorithms_and_parameters = [
            (it.Instance.static_CI_construction, dict()),
            (it.Instance.static_I1_construction, dict(init_method='earliest_due_date'))
        ]
        # create a centralized instance, i.e. only one carrier
        centralized_instance = base_instance.to_centralized(base_instance.carriers[0].depot.coords)
        for algorithm, parameters in algorithms_and_parameters:
            instance_copy = deepcopy(centralized_instance)
            algorithm(instance_copy, **parameters)
            instance_copy.write_solution_to_json()
            results.append(instance_copy.evaluation_metrics)
            # ====== 2 opt
            instance_copy.two_opt()
            instance_copy.write_solution_to_json()
            results.append(instance_copy.evaluation_metrics)

    return results


def multi_func(solomon, num_of_inst):
    """
    for the first num_of_inst random instances that exists in the solomon type folder, this function runs all available
    algorithms to solve each of these instances. The results are collected, transformed into a pd.DataFrame and saved as
    .csv
    :param solomon: solomon base class to be used. will read custom made classes from this base class
    :param num_of_inst: number of instances to read and solve
    """
    name = multiprocessing.current_process().name
    pid = os.getpid()
    print(f'{solomon} in {name} - {pid}')

    directory = path_input_custom.joinpath(solomon)
    inst_names = list(directory.iterdir())[:num_of_inst]  # find the first num_of_inst custom instances
    centralized_flag = True  # TRUE if also the centralized versions should be solved
    solomon_base_results = []  # collecting all the results
    for instance_name in tqdm(iterable=inst_names):
        path = directory.joinpath(instance_name)
        base_instance = it.read_custom_json_instance(path)
        results = run_all_algorithms(base_instance,
                                     centralized_flag)  # TODO: extract the creation of the eval.csv file!!
        solomon_base_results.extend(results)
        centralized_flag = False

    performance = pd.DataFrame(solomon_base_results)
    performance = performance.set_index(['solomon_base', 'rand_copy', 'algorithm', 'num_carriers', 'num_vehicles'])

    # write the results
    write_dir=path_output_custom.joinpath(solomon)
    write_dir.mkdir(parents=True, exist_ok=True)  # TODO All the directory creation should happen in the beginning somewhere once
    file_name = f'{base_instance.id_.split("#")[0]}eval.csv'
    write_path = write_dir.joinpath(file_name)
    performance.to_csv(write_path)


if __name__ == '__main__':
    # jobs = []
    # solomon_list = ['C101', 'C201', 'R101', 'R201', 'RC101', 'RC201']
    # for solomon in solomon_list:
    #     process = multiprocessing.Process(target=multi_func, args=(solomon, 100,))
    #     jobs.append(process)
    #     process.start()
    # for j in jobs:  # to ensure that program waits until all processes have finished before continuing
    #     j.join()

    # for running just a single instance
    multi_func('C101', num_of_inst=1)
