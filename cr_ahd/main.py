import datetime
import os
import json
from copy import deepcopy

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import instance as it
from utils import opts, Solomon_Instances, InsertionError


def main(path, centralized_flag):
    # read
    custom = it.read_custom_json_instance(path)
    if opts['plot_level'] > 0:
        custom.plot()
        plt.show()

    # ===== STATIC I1 INSERTION =====
    # Having the benefit of knowing all requests in advance, each carrier can use I1 insertion to always identify the
    # best unrouted customer for the next insertion
    sta_custom = deepcopy(custom)
    sta_custom.static_I1_construction()
    sta_custom.write_solution_to_json()

    # ===== DYNAMIC SEQUENTIAL INSERTION (NO AUCTION) =====
    dyn_custom = deepcopy(custom)
    dyn_custom.dynamic_construction(with_auction=False)
    dyn_custom.write_solution_to_json()

    # ===== DYNAMIC + AUCTION =====
    dyn_auc_custom = deepcopy(custom)
    dyn_auc_custom.dynamic_construction(with_auction=True)
    dyn_auc_custom.write_solution_to_json()

    # ===== CENTRALIZED  =====
    if centralized_flag:  # centralized instances need to be solved only once
        # ===== I1  =====
        cen_sta_I1_custom = deepcopy(custom).to_centralized(custom.carriers[0].depot.coords)
        cen_sta_I1_custom.static_I1_construction()
        cen_sta_I1_custom.write_solution_to_json()

        # ===== CHEAPEST INSERTION  =====
        cen_sta_cheapest_insertion_custom = deepcopy(custom).to_centralized(custom.carriers[0].depot.coords)
        cen_sta_cheapest_insertion_custom.static_CI_construction()
        cen_sta_cheapest_insertion_custom.write_solution_to_json()

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
    for solomon in ['C101', 'C201']: #, 'R101', 'R201', 'RC101', 'RC201']:
        directory = f'../data/Input/Custom/{solomon}'
        inst_names = os.listdir(directory)[:3]
        centralized_flag = False
        solomon_base_results = []
        for instance_name in tqdm(inst_names, ascii=True):
            path = os.path.join(directory, instance_name)
            results = main(path, centralized_flag)
            solomon_base_results.extend(results)
            centralized_flag = False

        performance = pd.DataFrame(solomon_base_results)
        performance = performance.set_index(['solomon_base', 'rand_copy', 'algorithm', 'num_carriers', 'num_vehicles'])
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        performance.to_csv(f'../data/Output/Custom/{solomon}/{timestamp}_eval.csv')

    # print(performance)
    # print(performance.describe())
    # performance.groupby(['solomon_base'])
    # performance.filter(regex='runtime').plot(marker='o', )
    # performance.filter(regex='cost').plot(marker='o')
    # plt.show()

    # TODO distribute different benchmarks/versions or different instances over multiple cores
    #  and remodel the evaluation: (1) save all results per instance (2) read every result file and store the relevant
    #  stuff in a df (3) make plots

    # TODO how do carriers 'buy' requests from others? Is there some kind of money exchange happening?

    # TODO storing the vehicle assignment with each vertex (also in the file) may greatly simplify a few things.
    #  Alternatively, store the assignment in the instance? Or have some kind of AssignmentManager class?!

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
