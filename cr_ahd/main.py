import datetime
from copy import deepcopy

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import instance as it
from utils import opts, Solomon_Instances, InsertionError


def main(solomon) -> dict:
    # create and write a new instance with random assignment based on a solomon instance
    num_carriers = 3
    num_vehicles = 20
    custom = it.make_custom_from_solomon(solomon,
                                         f'{solomon}_{num_carriers}_{num_vehicles}',
                                         num_carriers,
                                         num_vehicles,
                                         None)
    custom.assign_all_requests_randomly()
    # file_name = custom.write_custom_json()

    # Read file and create the instance
    # custom = read_custom_json(file_name)
    if opts['plot_level'] > 0:
        custom.plot()
        plt.show()

    # ===== DYNAMIC (NO AUCTION) =====
    dyn_runtime, dyn_cost, dyn_num_veh = deepcopy(custom).dynamic_construction(with_auction=False)

    # ===== DYNAMIC + AUCTION =====
    dyn_auc_runtime, dyn_auc_cost, dyn_auc_num_veh = deepcopy(custom).dynamic_construction(with_auction=True)

    # ===== STATIC CHEAPEST INSERTION =====
    sta_runtime, sta_cost, sta_num_veh = deepcopy(custom).static_construction(method='cheapest_insertion')

    # ===== CENTRALIZED STATIC CHEAPEST INSERTION =====
    # only sensible to repeatedly do this if the base instance (solomon) changes, otherwise I'd just repeat the same
    # computations
    try:
        centralized = custom.to_centralized()
        cen_sta_runtime, cen_sta_cost, cen_sta_num_veh = centralized.static_construction(method='cheapest_insertion')
    except InsertionError:
        cen_sta_runtime = None
        cen_sta_cost = None

    # result collection
    return dict(
        instance=custom.id_,
        dyn_runtime=dyn_runtime,
        dyn_cost=dyn_cost,
        dyn_auc_runtime=dyn_auc_runtime,
        dyn_auc_cost=dyn_auc_cost,
        sta_runtime=sta_runtime,
        sta_cost=sta_cost,
        cen_sta_runtime=cen_sta_runtime,
        cen_sta_cost=cen_sta_cost,
    )


if __name__ == '__main__':
    results = []
    # for i in tqdm(range(opts['num_trials']), ascii=True):
    for solomon in tqdm(Solomon_Instances[:3]):
        res = main(solomon)
        results.append(res)
    performance = pd.DataFrame(results)
    performance = performance.set_index('instance')
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    performance.to_csv(f'../data/Output/Performance{timestamp}.csv')
    print(performance)
    print(performance.describe())
    performance.filter(regex='runtime').plot(marker='o', )
    performance.filter(regex='cost').plot(marker='o')
    plt.show()

    # TODO how do carriers 'buy' requests from others? Is there some kind of money exchange happening?

    # TODO storing the vehicle assignment with each vertex (also in the file) may greatly simplify a few things.
    #  Alternatively, store the assignment in the instance?

    # TODO it seems that in the dynamic cases, ALL vehicles are tested for insertion while in the static one,
    #  only the current vehicle is checked -> Is that true? if yes, fix it

    # TODO distribute different benchmarks/versions over multiple cores

