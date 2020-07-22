import datetime
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import instance as it
from utils import opts, Solomon_Instances, InsertionError


def main(path, centralized_flag):

    # read
    custom = it.read_custom_json(path)
    if opts['plot_level'] > 0:
        custom.plot()
        plt.show()

    # ===== STATIC I1 INSERTION =====
    # Having the benefit of knowing all requests in advance, each carrier can use I1 insertion to always identify the
    # best unrouted customer for the next insertion
    sta_custom = deepcopy(custom)
    sta_custom.static_construction(method='I1')

    # ===== DYNAMIC SEQUENTIAL INSERTION (NO AUCTION) =====
    dyn_custom = deepcopy(custom)
    dyn_custom.dynamic_construction(with_auction=False)

    # ===== DYNAMIC + AUCTION =====
    dyn_auc_custom = deepcopy(custom)
    dyn_auc_custom.dynamic_construction(with_auction=True)

    # ===== CENTRALIZED  =====
    # if path == '../data/Custom/C101\\C101_3_15_ass_#000.json':
    if centralized_flag:
        # ===== I1  =====
        cen_sta_I1_custom = custom.to_centralized(custom.carriers[0].depot.coords)
        cen_sta_I1_custom.static_construction(method='I1', )

        # ===== CHEAPEST INSERTION  =====
        cen_sta_cheapest_insertion_custom = custom.to_centralized(custom.carriers[0].depot.coords)
        cen_sta_cheapest_insertion_custom.static_construction(method='cheapest_insertion', )

        # result collection
        return (sta_custom.evaluation_metrics,
                dyn_custom.evaluation_metrics,
                dyn_auc_custom.evaluation_metrics,
                cen_sta_I1_custom.evaluation_metrics,
                cen_sta_cheapest_insertion_custom.evaluation_metrics)
    else:
        return (sta_custom.evaluation_metrics,
                dyn_custom.evaluation_metrics,
                dyn_auc_custom.evaluation_metrics)
    # TODO extend metric collection: E.g. how many requests are finally assigned to each carrier?


if __name__ == '__main__':
    centralized_flag = False

    results = []
    directory = f'../data/Custom/C101'
    inst_names = os.listdir(directory)[:1]
    for instance_name in tqdm(inst_names):
        path = os.path.join(directory, instance_name)
        res = main(path, centralized_flag)
        centralized_flag = False
        results.extend(res)
    performance = pd.DataFrame(results)
    performance = performance.set_index(['id', 'algorithm'])
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    performance.to_csv(f'../data/Output/Performance{timestamp}.csv')
    print(performance)
    print(performance.describe())
    performance.filter(regex='runtime').plot(marker='o', )
    performance.filter(regex='cost').plot(marker='o')
    plt.show()

    # TODO how do carriers 'buy' requests from others? Is there some kind of money exchange happening?

    # TODO storing the vehicle assignment with each vertex (also in the file) may greatly simplify a few things.
    #  Alternatively, store the assignment in the instance? Or have some kind of AssignmentManager class?!

    # TODO distribute different benchmarks/versions or different instances over multiple cores

    # TODO how to handle initialization? initializing pendulum tours for all vehicles of a carrier is stupid as it
    #  opens up potentially unnecessary vehicles/tours. Is there a clever way to initialize a new tour only when
    #  necessary?

    # TODO re-integrate animated plots for
    #  (1) static/dynamic sequential cheapest insertion construction => DONE
    #  (2) dynamic construction
    #  (3) I1 insertion construction => DONE
