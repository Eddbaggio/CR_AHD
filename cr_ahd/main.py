import time

import matplotlib.pyplot as plt
import numpy as np

import instance as it
from utils import opts


def main() -> it.Instance:
    np.random.seed(3)
    # Read solomon file and create the instance
    C103: it.Instance = it.read_solomon('C103', num_carriers=3)
    fig, ax = plt.subplots()
    C103.plot(ax)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    plt.show()

    if opts['verbose'] > 1:
        print(C103.dist_matrix)

    # MAIN LOOP
    # randomly assign customers to carriers
    for r in C103.requests:
        c = C103.carriers[np.random.choice(range(3))]
        c.assign_request(r)

    if opts['verbose'] > 0:
        print(*C103.carriers, sep='\n')
        print('\n')
    C103.cheapest_insertion_construction()
    if opts['verbose'] > 0:
        print(f'Total cost: {C103.total_cost()}')
    return C103


if __name__ == '__main__':
    times = []
    cost = []
    for i in range(1):
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

    for c in inst.carriers:
        for v in c.vehicles:
            ax: plt.Axes
            fig, ax = plt.subplots()
            if len(v.tour) > 2:
                p = v.tour.plot(ax=ax)
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 100)
                plt.show()
