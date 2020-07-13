import time

import matplotlib.pyplot as plt
import numpy as np

import instance as it
from carrier import Carrier
from utils import opts, InsertionError


def main() -> it.Instance:
    np.random.seed(0)
    # Read solomon file and create the instance
    C101: it.Instance = it.read_solomon('C101', num_carriers=3)
    if opts['plot_level'] > 1:
        fig, ax = plt.subplots()
        C101.plot(ax)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        fig.show()

    if opts['verbose'] > 1:
        print(C101.dist_matrix)

    for u in C101.requests:
        if opts['verbose'] > 0:
            print('\n')
        initial_carrier = C101.carriers[np.random.choice(range(3))]
        # initial_carrier.assign_request(u)
        carrier, vehicle, position, cost = C101.cheapest_insertion_auction(u, initial_carrier)
        carrier.assign_request(u)

        # attempt insertion
        try:
            if opts['verbose'] > 0:
                print(f'\tInserting {u.id_} into {carrier.id_}.{vehicle.id_} with cost of {round(cost, 2)}')
            vehicle.tour.insert_and_reset_schedules(position, u)
            vehicle.tour.compute_cost_and_schedules(C101.dist_matrix)
            carrier.unrouted.pop(u.id_)  # remove inserted request from unrouted
        except TypeError:
            raise InsertionError('', f"Cannot insert {u} feasibly into {carrier.id_}.{vehicle.id_}")

    if opts['verbose'] > 0:
        print(*C101.carriers, sep='\n')
        print('\n')
        print(f'Total cost of {C101.id_}: {C101.total_cost()}')
    return C101


if __name__ == '__main__':
    times = []
    costs = []
    for i in range(opts['num_trials']):
        t0 = time.perf_counter()
        inst = main()
        t1 = time.perf_counter() - t0
        times.append(t1)
        costs.append(inst.total_cost())
    print(dict(iterations=i + 1,
               avg_cost=round(sum(costs) / len(costs), 4),
               min_cost=round(min(costs), 4),
               max_cost=round(max(costs), 4),
               avg_time=round(sum(times) / len(times), 4),
               min_time=round(min(times), 4),
               max_time=round(max(times), 4),
               ))

    if opts['plot_level'] > 0:
        c: Carrier
        for c in inst.carriers:
            p = c.plot()
            plt.xlim(0, 100)
            plt.ylim(0, 100)
            plt.title(f'tour {c.id_} with cost of {c.route_cost(2)}')
        plt.show()
