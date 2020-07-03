import time

import matplotlib.pyplot as plt
import numpy as np

import instance as it
from utils import opts


def main() -> it.Instance:
    # np.random.seed(3)
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

    # MAIN LOOP
    # randomly assign customers to carriers
    for r in C101.requests:
        c = C101.carriers[np.random.choice(range(3))]
        c.assign_request(r)

    if opts['verbose'] > 0:
        print(*C101.carriers, sep='\n')
        print('\n')
    C101.cheapest_insertion_construction()
    if opts['verbose'] > 0:
        print(f'Total cost: {C101.total_cost()}')
    return C101


if __name__ == '__main__':
    times = []
    cost = []
    for i in range(20):
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
        for c in inst.carriers:
            for v in c.vehicles:
                if len(v.tour) > 2:
                    fig: plt.Figure
                    ax: plt.Axes
                    fig, ax = plt.subplots()
                    p = v.tour.plot(ax=ax)
                    ax.set_xlim(0, 100)
                    ax.set_ylim(0, 100)
                    ax.set_title(f'tour {c.id_}.{v.id_} with cost of {c.route_cost(2)}')
                    fig.show()
                    plt.close(fig)
