import instance as it

import numpy as np


if __name__ == '__main__':
    # Read solomon file and create the instance
    C101 = it.read_solomon('C101', num_carriers=3)
    print(C101.dist_matrix)

    # MAIN LOOP
    # assign customer to carrier randomly
    for r in C101.requests:
        c = C101.carriers[np.random.choice(range(3))]
        c.assign_request(r)

    print(*C101.carriers, sep='\n')
