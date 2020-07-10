import time

import matplotlib.pyplot as plt
import numpy as np

import instance as it
from carrier import Carrier
from utils import opts


def main() -> it.Instance:
    np.random.seed(0)
    # Read solomon file and create the instance
    C101: it.Instance = it.read_solomon('C101', num_carriers=3)
    if opts['plot_level'] > 1:
        C101.plot()
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.show()

    if opts['verbose'] > 1:
        print(C101.dist_matrix)

    # assign requests to carriers randomly
    C101.assign_all_requests()

    # construct initial solution
    C101.static_construction(method='cheapest_insertion')

    if opts['verbose'] > 0:
        print(*C101.carriers, sep='\n')
        print('\n')
        print(f'Total cost of {C101.id_}: {C101.total_cost()}')
    return C101


if __name__ == '__main__':
    times = []
    cost = []
    for i in range(opts['num_trials']):
        t0 = time.perf_counter()
        inst = main()
        t1 = time.perf_counter() - t0
        times.append(t1)
        cost.append(inst.total_cost())
    print(dict(iterations=i + 1,
               avg_cost=round(sum(cost) / len(cost), 4),
               min_cost=round(min(cost), 4),
               max_cost=round(max(cost), 4),
               avg_time=round(sum(times) / len(times), 4),
               min_time=round(min(times), 4),
               max_time=round(max(times), 4),
               ))

    if opts['plot_level'] > 0:
        c: Carrier
        for c in inst.carriers:
            p = c.plot()
        plt.show()
