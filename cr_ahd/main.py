import instance as it
from utils import opts

import numpy as np


if __name__ == '__main__':
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

    print(*C101.carriers, sep='\n')
    print('\n')
    C101.cheapest_insertion_construction()
    print(f'Total cost: {C101.total_cost()}')

    # TODO: I1_construction is acutally chepaest insertion atm; tours are not yet stored in vehicle or carrier class
