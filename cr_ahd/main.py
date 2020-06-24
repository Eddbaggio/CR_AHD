import instance as it
from utils import opts
import numpy as np
import time
from tqdm import tqdm


def main():
    np.random.seed(3)
    # Read solomon file and create the instance
    C101 = it.read_solomon('C101', num_carriers=3)
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
    C101.I1_construction()
    if opts['verbose'] > 0:
        print(f'Total cost: {C101.total_cost()}')
    return C101.total_cost()


if __name__ == '__main__':
    times = []
    cost = []
    for i in range(10):
        t0 = time.perf_counter()
        c = main()
        t1 = time.perf_counter() - t0
        times.append(t1)
        cost.append(c)
    print(dict(iterations=i + 1,
               avg_time=round(sum(times) / len(times), 4),
               min_time=round(min(times), 4),
               max_time=round(max(times), 4),
               avg_cost=round(sum(cost) / len(cost), 4),
               min_cost=round(min(cost), 4),
               max_cost=round(max(cost), 4)))
