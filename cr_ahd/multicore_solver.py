import multiprocessing
import os
from copy import deepcopy

import pandas as pd

import instance as it


def run_all_algorithms(base_instance: it.Instance, centralized_flag: bool):
    results = []

    algorithms_and_parameters = [
        (it.Instance.static_CI_construction, dict()),
        (it.Instance.dynamic_construction, dict(with_auction=False)),
        (it.Instance.dynamic_construction, dict(with_auction=True))
    ]
    for algorithm, parameters in algorithms_and_parameters:
        instance_copy = deepcopy(base_instance)
        algorithm(instance_copy, **parameters)
        instance_copy.write_solution_to_json()
        results.append(instance_copy.evaluation_metrics)

    # centralized instances
    if centralized_flag:
        algorithms_and_parameters = [
            (it.Instance.static_CI_construction, dict()),
            (it.Instance.static_I1_construction, dict(init_method='earliest_due_date'))
        ]
        centralized_instance = base_instance.to_centralized(base_instance.carriers[0].depot.coords)
        for algorithm, parameters in algorithms_and_parameters:
            instance_copy = deepcopy(centralized_instance)
            algorithm(instance_copy, **parameters)
            instance_copy.write_solution_to_json()
            results.append(instance_copy.evaluation_metrics)

    return results


def multi_func(solomon, num_of_inst):
    directory = f'../data/Input/Custom/{solomon}'
    inst_names = os.listdir(directory)[:num_of_inst]
    centralized_flag = True
    solomon_base_results = []
    for instance_name in inst_names:
        path = os.path.join(directory, instance_name)
        base_instance = it.read_custom_json_instance(path)
        results = run_all_algorithms(base_instance, centralized_flag)
        solomon_base_results.extend(results)
        centralized_flag = False

    performance = pd.DataFrame(solomon_base_results)
    performance = performance.set_index(['solomon_base', 'rand_copy', 'algorithm', 'num_carriers', 'num_vehicles'])
    file_name = f'{base_instance.id_.split("#")[0]}eval.csv'
    performance.to_csv(f'../data/Output/Custom/{solomon}/{file_name}')

    name = multiprocessing.current_process().name
    pid = os.getpid()
    print(f'{solomon} in {name} - {pid}')


if __name__ == '__main__':
    jobs = []
    for solomon in ['C101', 'C201', 'R101', 'R201', 'RC101', 'RC201']:
        process = multiprocessing.Process(target=multi_func, args=(solomon, 25,))
        jobs.append(process)
        process.start()
    # multi_func('C101', 2)
