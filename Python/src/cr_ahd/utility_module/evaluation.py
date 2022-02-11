from pprint import pprint

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

instance_config = [
    't',
    'd',
    'c',
    'n',
    'v',
    'o',
    'r',
]
planning_config = 'solution_algorithm'
general_config = [
    'tour_improvement',
    'neighborhoods',
    'tour_construction',
    'tour_improvement_time_limit_per_carrier',
    'max_num_accepted_infeasible',
    'request_acceptance_attractiveness',
    'time_window_length',
    'time_window_offering',
    'time_window_selection',
]
collaborative_config = [
    'num_int_auctions',
    'int_auction_tour_construction',
    'int_auction_tour_improvement',
    'int_auction_neighborhoods',
    'int_auction_num_submitted_requests',
    'int_auction_request_selection',
    'int_auction_bundle_generation',
    'int_auction_partition_valuation',
    'int_auction_num_auction_bundles',
    'int_auction_bidding',
    'int_auction_winner_determination',
    'int_auction_num_auction_rounds',
    'fin_auction_tour_construction',
    'fin_auction_tour_improvement',
    'fin_auction_neighborhoods',
    'fin_auction_num_submitted_requests',
    'fin_auction_request_selection',
    'fin_auction_bundle_generation',
    'fin_auction_partition_valuation',
    'fin_auction_num_auction_bundles',
    'fin_auction_bidding',
    'fin_auction_winner_determination',
    'fin_auction_num_auction_rounds',
]
solution_values = [
    'objective',
    # 'sum_profit',
    'sum_travel_distance',
    'sum_travel_duration',
    'sum_load',
    'sum_revenue',
    'num_tours',
    'num_pendulum_tours',
    'num_routing_stops',
    'acceptance_rate',
    'degree_of_reallocation',
]
solution_runtimes = [
    'runtime_final_improvement',
    'runtime_total',
    'runtime_final_auction',
]


def collaboration_gain(df: pd.DataFrame, plot: bool = False):
    gains = []
    variable_parameters = {k: list(df[k].dropna().unique()) for k in
                           instance_config + general_config + collaborative_config}
    variable_parameters = {k: v for k, v in variable_parameters.items() if len(v) > 1}
    print()
    print('Variable Parameters'.center(50, '='))
    pprint(variable_parameters, sort_dicts=False)

    # group/filter by instance type (num_requests, overlap, ...)
    for inst_name, inst_group in df.groupby(instance_config):
        # group/filter by general solver parameters (time_window_length, tour_construction, ...)
        for gen_name, gen_group in inst_group.groupby(general_config, dropna=False):
            isolated = gen_group[gen_group[planning_config] == 'IsolatedPlanning'].squeeze()
            collaborative = gen_group[gen_group[planning_config] == 'CollaborativePlanning'].squeeze()
            if isinstance(collaborative, pd.Series):
                coll_gain = 1 - (collaborative[solution_values] / isolated[solution_values])
                record = {k: v for k, v in zip(instance_config + general_config, inst_name)}
                record.update({k: v for k, v in zip(general_config, gen_name)})
                record.update({k: v for k, v in zip(solution_values, coll_gain)})
                gains.append(record)
            else:
                # group/filter by collaborative parameters (num_submitted_requests, bundle_valuation, ...)
                for collab_name, collab_group in collaborative.groupby(collaborative_config, dropna=False):
                    collab_group = collab_group.squeeze()
                    coll_gain = 1 - (collab_group[solution_values] / isolated[solution_values])
                    record = {k: v for k, v in zip(instance_config, inst_name)}
                    record.update({k: v for k, v in zip(general_config, gen_name)})
                    record.update({k: v for k, v in zip(collaborative_config, collab_name)})
                    record.update({k: v for k, v in zip(solution_values, coll_gain)})
                    gains.append(record)

    df = pd.DataFrame.from_records(gains,
                                   columns=instance_config + general_config + collaborative_config + solution_values)
    df.set_index(instance_config + general_config + collaborative_config, inplace=True)
    if plot:
        df['objective'].plot(kind='bar').legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5)
        plt.show()
    return df


if __name__ == '__main__':
    # path = "C:/Users/Elting/Desktop/HPC_Output/evaluation_agg_solution_#011.csv"
    path = "C:/Users/Elting/ucloud/PhD/02_Research/02_Collaborative Routing for Attended Home Deliveries/01_Code/data/Output/evaluation_agg_solution_#021.csv"
    df = pd.read_csv(path)

    collaboration_gain(df, True)
    # .to_csv(path.replace('agg_solution', 'coll_gain'))
