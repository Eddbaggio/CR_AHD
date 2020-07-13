import json
import time
from copy import copy
from itertools import islice
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import carrier as cr
import vehicle as vh
import vertex as vx
from utils import split_iterable, make_dist_matrix, opts, InsertionError


class Instance(object):
    """Class to store CR_AHD instances
    """

    def __init__(self, id_: str, requests: List[vx.Vertex], carriers: List[cr.Carrier],
                 dist_matrix: pd.DataFrame = None):
        self.id_ = id_
        self.requests = requests
        self.carriers = carriers
        if dist_matrix is not None:
            self.dist_matrix = dist_matrix
        else:
            self.dist_matrix = make_dist_matrix([*self.requests, *[c.depot for c in self.carriers]])
        pass

    def __str__(self):
        return f'Instance {self.id_} with {len(self.requests)} customers and {len(self.carriers)} carriers'

    def to_dict(self):
        return {
            'id_': self.id_,
            'requests': [r.to_dict() for r in self.requests],
            'carriers': [c.to_dict() for c in self.carriers],
            'dist_matrix': self.dist_matrix.to_dict()
        }

    def assign_all_requests(self):
        for r in self.requests:
            c = self.carriers[np.random.choice(range(3))]
            c.assign_request(r)

    def static_construction(self, method: str, verbose: int = opts['verbose']):
        assert method in ['cheapest_insertion', 'I1']

        t0 = time.perf_counter()

        if verbose > 0:
            print(f'STATIC {method} Construction:')
        for c in self.carriers:
            if method == 'cheapest_insertion':
                c.static_cheapest_insertion_construction(self.dist_matrix)
            elif method == 'I1':
                c.static_I1_construction(dist_matrix=self.dist_matrix)
            if verbose > 0:
                print(f'Total Route cost of carrier {c.id_}: {c.route_cost()}\n')

        runtime = time.perf_counter() - t0
        return runtime, self.total_cost()



    def cheapest_insertion_auction(self, request: vx.Vertex, initial_carrier: cr.Carrier, verbose=opts['verbose'],
                                   plot_level=opts['plot_level']):
        """
        If insertion of the request is profitable (demand > insertion cost) for the initial carrier, returns the
        <carrier, vehicle, position, cost> triple for the cheapest insertion. If insertion is not profitable for the
        initially selected carrier, returns the associated triple for the collaborator with the cheapest insertion
        cost for the given request

        :return: The carrier, vehicle, insertion index, and insertion cost of the correct/best insertion
        """
        if verbose > 0:
            print(f'{request.id_} is originally assigned to {initial_carrier.id_}')
            print(f'Checking profitability of {request.id_} for {initial_carrier.id_}')
        vehicle_best, position_best, cost_best = initial_carrier.find_cheapest_feasible_insertion(request,
                                                                                                  self.dist_matrix)
        carrier_best = initial_carrier
        # auction: determine and return the collaborator with the cheapest insertion cost
        if cost_best > request.demand:
            if verbose > 0:
                print(f'{request.id_} is not profitable for {carrier_best.id_}')
            for carrier in self.carriers:
                if carrier is carrier_best:
                    continue
                else:
                    if verbose > 0:
                        print(f'Checking profitability of {request.id_} for {carrier.id_}')
                    vehicle, position, cost = carrier.find_cheapest_feasible_insertion(request, self.dist_matrix)
                    if cost < cost_best:
                        carrier_best = carrier
                        vehicle_best = vehicle
                        position_best = position
                        cost_best = cost
        if verbose > 0:
            if cost_best > request.demand:
                print(
                    f'No carrier can insert request {request.id_} profitably! It will be assigned to {carrier_best.id_}')
            else:
                print(f'{request.id_} is finally assigned to {carrier_best.id_}')
        return carrier_best, vehicle_best, position_best, cost_best

        # TODO: continue here! request are assigned one by one and then the cheapest feasible insertion cost are
        #  determined. Take into account the demand value of a request for the cheapest insertion. This will allow to
        #  determine whether a request is profitable or not. If it is not profitable, submit is to the auction

    def dynamic_construction(self):
        t0 = time.perf_counter()
        for i in range(len(self.requests)):
            # find the next request u, that has id number i
            for c in self.carriers:
                try:
                    u_id, u = next(islice(c.unrouted.items(), 1))  # get the first unrouted request of carrier c
                except StopIteration:
                    pass
                if int(u_id[1:]) == i:
                    break

            # do the auction
            carrier, vehicle, position, cost = self.cheapest_insertion_auction(u, c)
            if c != carrier:
                c.unrouted.pop(u.id_)  # remove from initial carrier
                c.requests.pop(u.id_)
                carrier.assign_request(u)  # assign to auction winner

            # attempt insertion
            try:
                if opts['verbose'] > 0:
                    print(f'\tInserting {u.id_} into {carrier.id_}.{vehicle.id_} with cost of {round(cost, 2)}')
                vehicle.tour.insert_and_reset_schedules(position, u)
                vehicle.tour.compute_cost_and_schedules(self.dist_matrix)
                carrier.unrouted.pop(u.id_)  # remove inserted request from unrouted
            except TypeError:
                raise InsertionError('', f"Cannot insert {u} feasibly into {carrier.id_}.{vehicle.id_}")

        runtime = time.perf_counter() - t0
        return runtime, self.total_cost()

    def total_cost(self):
        total_cost = 0
        for c in self.carriers:
            for v in c.vehicles:
                total_cost += v.tour.cost
        return total_cost

    def plot(self, annotate: bool = True, alpha: float = 1):
        # plot depots
        depots = [c.depot for c in self.carriers]
        for d in depots:
            plt.scatter(d.coords.x, d.coords.y, marker='s', alpha=alpha)
            if annotate:
                plt.annotate(f'{d.id_}', xy=d.coords)

        # plot requests locations
        for r in self.requests:
            plt.scatter(r.coords.x, r.coords.y, alpha=alpha, color='grey')
            if annotate:
                plt.annotate(f'{r.id_}', xy=r.coords)
        return

    def write_custom_json(self):
        assigned_requests = []
        for c in self.carriers:
            assigned_requests.extend(c.requests)
        if any(assigned_requests):
            file_name = f'../data/Custom/{self.id_}_assigned.json'
        else:
            file_name = f'../data/Custom/{self.id_}.json'
        with open(file_name, mode='w') as write_file:
            json.dump(self.to_dict(), write_file, indent=4)


def read_solomon(name: str, num_carriers: int) -> Instance:
    # TODO: a more efficient way of reading the data, e.g. by reading all vertices at once and split off the depot
    #  after; and by not using pd to read vehicles
    vehicles = pd.read_csv(f'../data/Solomon/{name}.txt', skiprows=4, nrows=1, delim_whitespace=True,
                           names=['number', 'capacity'])
    vehicles = [vh.Vehicle('v' + str(i), vehicles.capacity[0]) for i in range(vehicles.number[0])]
    vehicles = split_iterable(vehicles, num_carriers)

    cols = ['cust_no', 'x_coord', 'y_coord', 'demand', 'ready_time', 'due_date', 'service_duration']
    requests = pd.read_csv(f'../data/Solomon/{name}.txt', skiprows=10, delim_whitespace=True, names=cols)
    requests = [
        vx.Vertex('r' + str(row.cust_no - 1), row.x_coord, row.y_coord, row.demand, row.ready_time, row.due_date) for
        row in requests.itertuples()]

    depot = pd.read_csv(f'../data/Solomon/{name}.txt', skiprows=9, nrows=1, delim_whitespace=True,
                        names=cols)  # TODO: how to handle depots? read in read_solomon?
    depot = vx.Vertex('d1', int(depot.x_coord[0]), int(depot.y_coord[0]), int(depot.demand[0]),
                      int(depot.ready_time[0]),
                      int(depot.due_date[0]))
    carriers = []
    for i in range(num_carriers):
        c = cr.Carrier('c' + str(i), copy(depot), next(vehicles))
        c.depot.id_ = 'd' + str(i)
        carriers.append(c)
    inst = Instance(name, requests, carriers)
    if opts['verbose'] > 0:
        print(f'Successfully read Solomon {name} with {num_carriers} carriers.')
        print(inst)
    return inst


def make_custom_from_solomon(solomon_name: str, custom_name: str, num_carriers: int, num_vehicles_per_carrier: int,
                             vehicle_capacity: int, verbose: int = opts['verbose']):
    solomon = read_solomon(solomon_name, num_carriers)
    carriers = []
    for i in range(num_carriers):
        vehicles = []
        for j in range(num_vehicles_per_carrier):
            v = vh.Vehicle('v' + str(i * num_vehicles_per_carrier + j),
                           vehicle_capacity)
            vehicles.append(v)
        carrier = cr.Carrier('c' + str(i),
                             depot=solomon.carriers[i].depot,
                             vehicles=vehicles)
        carriers.append(carrier)
    inst = Instance(custom_name, solomon.requests, carriers)
    if verbose > 0:
        print(f'Created custom instance {inst}')
    return inst


def read_custom_json(name: str):
    file_name = f'../data/Custom/{name}.json'
    with open(file_name, 'r') as reader_file:
        json_data = json.load(reader_file)
    requests = [vx.Vertex(**r_dict) for r_dict in json_data['requests']]
    carriers = []
    for c_dict in json_data['carriers']:
        depot = vx.Vertex(**c_dict['depot'])
        vehicles = [vh.Vehicle(**v_dict) for v_dict in c_dict['vehicles']]
        assigned_requests = {r_dict['id_']: vx.Vertex(**r_dict) for r_dict in c_dict['requests']}
        c = cr.Carrier(c_dict['id_'], depot, vehicles, requests=assigned_requests, unrouted=assigned_requests)
        carriers.append(c)
    dist_matrix = pd.DataFrame.from_dict(json_data['dist_matrix'])
    inst = Instance(json_data['id_'], requests, carriers, dist_matrix)
    return inst


if __name__ == '__main__':
    num_carriers = 3
    num_vehicles = 10
    self = make_custom_from_solomon('R101',
                                      f'R101_{num_carriers}_{num_vehicles}',
                                    num_carriers,
                                    num_vehicles,
                                    None)
    self.assign_all_requests()
    self.write_custom_json()
    # custom = read_custom_json(f'R101_{num_carriers}_{num_vehicles}')
    pass
