from typing import Tuple, Any

import matplotlib.pyplot as plt

import instance as it
from utils import opts
import pandas as pd
from copy import deepcopy
from tqdm import tqdm


def main() -> Tuple[Any, Any, Any, Any, Any]:
    # create and write a new instance with random assignment based on a solomon instance
    num_carriers = 3
    num_vehicles = 10
    solomon = 'C101'
    custom = it.make_custom_from_solomon(solomon,
                                         f'{solomon}_{num_carriers}_{num_vehicles}',
                                         num_carriers,
                                         num_vehicles,
                                         None)
    custom.assign_all_requests()
    # TODO in order to save them to disk I must add an enumeration to the file names
    # custom.write_custom_json()

    # Read file and create the instance
    # custom = read_custom_json(f'{solomon}_{num_carriers}_{num_vehicles}')
    if opts['plot_level'] > 0:
        custom.plot()
        plt.show()

    # ===== DYNAMIC + AUCTION =====
    dyn_runtime, dyn_cost = deepcopy(custom).dynamic_construction()

    # ===== STATIC CHEAPEST INSERTION =====
    sta_runtime, sta_cost = deepcopy(custom).static_construction(method='cheapest_insertion')
    return custom, dyn_runtime, dyn_cost, sta_runtime, sta_cost


if __name__ == '__main__':
    instances = []
    dyn_runtimes = []
    dyn_costs = []
    sta_runtimes = []
    sta_costs = []
    for i in tqdm(range(opts['num_trials']), ascii=True):
        inst, dyn_runtime, dyn_cost, sta_runtime, sta_cost = main()
        instances.append(inst)
        dyn_runtimes.append(dyn_runtime)
        dyn_costs.append(dyn_cost)
        sta_runtimes.append(sta_runtime)
        sta_costs.append(sta_cost)
    performance = pd.DataFrame({'dyn_runtimes': dyn_runtimes,
                                'dyn_costs': dyn_costs,
                                'sta_runtimes': sta_runtimes,
                                'sta_costs': sta_costs},
                               index=[inst.id_ for inst in instances])
    print(performance)
    print(performance.describe())
    performance.filter(regex='runtimes').plot()
    performance.filter(regex='costs').plot()
    plt.show()
