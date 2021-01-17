import time

import matplotlib.pyplot as plt

import instance as it
from carrier import Carrier
from utils import opts


# TODO: describe what this file is for

def main() -> it.Instance:
    # Read file and create the instance
    C101_3_10_assigned: it.Instance = it.read_custom_json_instance('C101_3_10_assigned')  # TODO *poath certainly not working
    if opts['plot_level'] > 1:
        C101_3_10_assigned.plot()
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.show()

    if opts['verbose'] > 1:
        print(C101_3_10_assigned.dist_matrix)

    # construct initial solution
    C101_3_10_assigned.static_CI_construction(method='cheapest_insertion')
    # C101_3_10_assigned.static_construction(method='I1')

    if opts['verbose'] > 0:
        print(*C101_3_10_assigned.carriers, sep='\n')
        print('\n')
        print(f'Total cost of {C101_3_10_assigned.id_}: {C101_3_10_assigned.cost}')
    return C101_3_10_assigned


if __name__ == '__main__':
    times = []
    cost = []
    for i in range(opts['num_trials']):
        t0 = time.perf_counter()
        inst = main()
        t1 = time.perf_counter() - t0
        times.append(t1)
        cost.append(inst.cost)
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
