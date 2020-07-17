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
    # custom = read_custom_json(file_name)
    if opts['plot_level'] > 0:
        custom.plot()
        plt.show()

    # ===== STATIC CHEAPEST INSERTION =====
    # Since this uses a SEQUENTIAL insertion strategy (inserting unrouted customers in the order that they were
    #  assigned), the result is the same as the dynamic insertion!

    # sta_custom = deepcopy(custom)
    # _, sta_runtime = sta_custom.static_construction(method='cheapest_insertion')
    # sta_cost = sta_custom.total_cost
    # sta_solution = sta_custom.solution
    # print(sta_solution)

    # ===== STATIC I1 INSERTION =====
    # Having the benefit of knowing all requests in advance, each carrier can use I1 insertion to always identify the
    # best unrouted customer for the next insertion

    # ===== DYNAMIC (NO AUCTION) =====
    dyn_custom = deepcopy(custom)
    _, dyn_runtime = dyn_custom.dynamic_construction(with_auction=False)
    dyn_cost = dyn_custom.total_cost
    dyn_solution = dyn_custom.solution
    # print(dyn_solution)

    # ===== DYNAMIC + AUCTION =====
    dyn_auc_custom = deepcopy(custom)
    _, dyn_auc_runtime = dyn_auc_custom.dynamic_construction(with_auction=True)
    dyn_auc_cost = dyn_auc_custom.total_cost

    # ===== CENTRALIZED STATIC/DYNAMIC CHEAPEST INSERTION =====
    # TODO There is no difference in static and dynamic if i use SEQUENTIAL cheapest insertion!
    try:
        centralized = custom.to_centralized(custom.carriers[0].depot.coords)
        _, cen_sta_runtime = centralized.static_construction(method='cheapest_insertion')
        cen_sta_cost = centralized.total_cost
    except InsertionError:
        cen_sta_runtime = None
        cen_sta_cost = None

    # result collection
    return dict(
        instance=custom.id_,
        # sta_cost=sta_cost,
        dyn_cost=dyn_cost,
        dyn_auc_cost=dyn_auc_cost,
        cen_sta_cost=cen_sta_cost,
        # sta_runtime=sta_runtime,
        dyn_runtime=dyn_runtime,
        dyn_auc_runtime=dyn_auc_runtime,
        cen_sta_runtime=cen_sta_runtime,
    )


if __name__ == '__main__':
    results = []
    # for i in tqdm(range(opts['num_trials']), ascii=True):
    for solomon in tqdm([Solomon_Instances[0]]):
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

    # TODO distribute different benchmarks/versions over multiple cores

    # TODO update/fix/re-integrate the I1 insertion. Originally does not check every vehicle for insertion. Do I want to
    #  stick to the original version?

    # TODO re-integrate animated plots for (1) static/dynamic sequential cheapest insertion construction (2) dynamic
    #  construction (3) I1 insertion construction
