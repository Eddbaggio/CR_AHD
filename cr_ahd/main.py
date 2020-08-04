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
    sta_custom.static_I1_construction(init_method='earliest_due_date')

    # ===== DYNAMIC SEQUENTIAL INSERTION (NO AUCTION) =====
    dyn_custom = deepcopy(custom)
    dyn_custom.dynamic_construction(with_auction=False)

    # ===== DYNAMIC + AUCTION =====
    dyn_auc_custom = deepcopy(custom)
    dyn_auc_custom.dynamic_construction(with_auction=True)

    # ===== CENTRALIZED  =====
    if centralized_flag:
        # ===== I1  =====
        cen_sta_I1_custom = deepcopy(custom).to_centralized(custom.carriers[0].depot.coords)
        cen_sta_I1_custom.static_I1_construction(init_method='earliest_due_date')

        # ===== CHEAPEST INSERTION  =====
        cen_sta_cheapest_insertion_custom = deepcopy(custom).to_centralized(custom.carriers[0].depot.coords)
        cen_sta_cheapest_insertion_custom.static_CI_construction()

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
    # TODO extend metric collection:
    #  - bidding price per request to identify/learn which requests are valuable for which coll. partner?
    #  - ... ?


if __name__ == '__main__':
    for solomon in ['C101', 'C201', 'R101', 'R201', 'RC101', 'RC201']:
        directory = f'../data/Custom/{solomon}'
        inst_names = os.listdir(directory)  # [:10]
        results = []
        centralized_flag = False
        for instance_name in tqdm(inst_names[:2], ascii=True):
            path = os.path.join(directory, instance_name)
            res = main(path, centralized_flag)
            centralized_flag = False
            results.extend(res)
        performance = pd.DataFrame(results)
        performance = performance.set_index(['id', 'num_carriers', 'num_vehicles', 'algorithm'])
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        performance.to_csv(f'../data/Output/Prototype/{solomon}_Performance{timestamp}.csv')
        print(performance)
        print(performance.describe())
        performance.groupby(['algorithm'])
        # performance.filter(regex='runtime').plot(marker='o', )
        # performance.filter(regex='cost').plot(marker='o')
        # plt.show()

    # TODO how do carriers 'buy' requests from others? Is there some kind of money exchange happening?

    # TODO storing the vehicle assignment with each vertex (also in the file) may greatly simplify a few things.
    #  Alternatively, store the assignment in the instance? Or have some kind of AssignmentManager class?!

    # TODO distribute different benchmarks/versions or different instances over multiple cores

    # TODO how to handle initialization? initializing pendulum tours for all vehicles of a carrier is stupid as it
    #  opens up potentially unnecessary vehicles/tours. Is there a clever way to initialize a new tour only when
    #  necessary?

    # TODO which of the @properties should be converted to proper class attributes, i.e. without delaying their
    #  computation? the @property may slow down the code, BUT in many cases it's probably a more idiot-proof way
    #  because otherwise I'd have to update the attribute which can easily be forgotten

    # TODO re-integrate animated plots for
    #  (1) static/dynamic sequential cheapest insertion construction => DONE
    #  (2) dynamic construction
    #  (3) I1 insertion construction => DONE
