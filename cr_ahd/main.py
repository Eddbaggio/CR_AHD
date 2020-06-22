import instance as it
from utils import opts
import numpy as np
import time
from tqdm import tqdm


def main():
    # np.random.seed(3)
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
    C101.cheapest_insertion_construction()
    if opts['verbose'] > 0:
        print(f'Total cost: {C101.total_cost()}')

    # TODO: I1_construction


if __name__ == '__main__':
    times = []
    for i in tqdm(range(100), ascii=True):
        t0 = time.perf_counter()
        main()
        t1 = time.perf_counter() - t0
        times.append(t1)
    print(dict(iterations=i + 1,
               avg_time=sum(times) / len(times),
               min_time=min(times),
               max_time=max(times)))
