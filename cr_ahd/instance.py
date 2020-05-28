import request as rq
import vehicle as vh
import carrier as cr
from utils import split_iterable, make_dist_matrix

import pandas as pd


class Instance(object):
    """Class to store CR_AHD instances


    """

    def __init__(self, id_, requests, carriers):
        self.id_ = id_
        self.requests = requests
        self.carriers = carriers
        self.dist_matrix = make_dist_matrix([*self.requests, *[c.depot for c in self.carriers]])
        pass

    def __str__(self):
        return f'Instance with {len(self.requests)} customers and {len(self.carriers)} carriers'


def read_solomon(name: str, num_carriers: int = 3):
    # TODO: read file in one go an split it afterwards
    cols = ['cust_no', 'x_coord', 'y_coord', 'demand', 'ready_time', 'due_date', 'service_time']
    requests = pd.read_csv(f'../data/Solomon/{name}.txt', skiprows=10, delim_whitespace=True, names=cols)
    requests = [rq.Request('r' + str(row.cust_no), row.x_coord, row.y_coord, row.ready_time, row.due_date) for row in requests.itertuples()]

    vehicles = pd.read_csv(f'../data/Solomon/{name}.txt', skiprows=4, nrows=1, delim_whitespace=True, names=['number', 'capacity'])
    vehicles = [vh.Vehicle('v' + str(i), vehicles.capacity[0]) for i in range(vehicles.number[0])]

    depot = pd.read_csv(f'../data/Solomon/{name}.txt', skiprows=8, nrows=1, delim_whitespace=True, names=cols)  # TODO: how to handle depots? read in read_solomon?
    depot = rq.Request('d1', depot.x_coord[0], depot.y_coord[0], depot.ready_time[0], depot.due_date[0])

    carriers = []
    vehicles = split_iterable(vehicles, num_carriers)
    for i in range(num_carriers):
        c = cr.Carrier('c' + str(i), depot, next(vehicles))
        carriers.append(c)

    return Instance(name, requests, carriers)
