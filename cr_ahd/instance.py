import json
import os
import time
from copy import copy
from itertools import islice
from typing import List

import matplotlib.pyplot as plt
from adjustText import adjust_text
import numpy as np
import pandas as pd

import carrier as cr
import vehicle as vh
import vertex as vx
from utils import split_iterable, make_dist_matrix, opts, InsertionError, timer


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

    @property
    def total_cost(self):
        total_cost = 0
        for c in self.carriers:
            for v in c.vehicles:
                total_cost += v.tour.cost
        return total_cost

    @property
    def num_vehicles_in_use(self):
        num_vehicles_in_use = 0
        for c in self.carriers:
            for v in c.vehicles:
                if len(v.tour) > 2:
                    num_vehicles_in_use += 1
        return num_vehicles_in_use

    def to_dict(self):
        return {
            'id_': self.id_,
            'requests': [r.to_dict() for r in self.requests],
            'carriers': [c.to_dict() for c in self.carriers],
            'dist_matrix': self.dist_matrix.to_dict()
        }

    def assign_all_requests_randomly(self):
        # np.random.seed(0)
        for r in self.requests:
            c = self.carriers[np.random.choice(range(len(self.carriers)))]
            c.assign_request(r)

    @ timer
    def static_construction(self, method: str, verbose: int = opts['verbose'], plot_level: int = opts['plot_level']):
        assert method in ['cheapest_insertion', 'I1']
        if verbose > 0:
            print(f'STATIC {method} Construction:')

        for c in self.carriers:
            if method == 'cheapest_insertion':
                while len(c.unrouted) > 0:
                    _, u = c.unrouted.popitem(last=False)
                    vehicle, position, cost = c.cheapest_feasible_insertion(u, self.dist_matrix, verbose, plot_level)
                    if verbose > 0:
                        print(f'\tInserting {u.id_} into {c.id_}.{vehicle.tour.id_}')
                    vehicle.tour.insert_and_reset_schedules(index=position, vertex=u)
                    vehicle.tour.compute_cost_and_schedules(self.dist_matrix)

            # elif method == 'I1':
            #     c.static_I1_construction(self.dist_matrix, verbose, plot_level)
            if verbose > 0:
                print(f'Total Route cost of carrier {c.id_}: {c.route_cost()}\n')
        return

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
        vehicle_best, position_best, cost_best = initial_carrier.cheapest_feasible_insertion(request,
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
                    vehicle, position, cost = carrier.cheapest_feasible_insertion(request, self.dist_matrix)
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

    @timer
    def dynamic_construction(self, with_auction: bool = True):
        # find the next request u, that has id number i
        # TODO this can be simplified/made more efficient if the
        #  assignment of a vertex is stored with its class instance. In that case, it must also be stored
        #  accordingly in the json file

        for i in range(len(self.requests)):
            for c in self.carriers:
                try:
                    u_id, u = next(islice(c.unrouted.items(), 1))  # get the first unrouted request of carrier c
                except StopIteration:
                    pass
                if int(u_id[1:]) == i:
                    break

            if with_auction:
                # do the auction
                carrier, vehicle, position, cost = self.cheapest_insertion_auction(request=u, initial_carrier=c)
                if c != carrier:
                    c.unrouted.pop(u.id_)  # remove from initial carrier
                    c.requests.pop(u.id_)
                    carrier.assign_request(u)  # assign to auction winner
            else:
                # find cheapest insertion
                carrier = c
                vehicle, position, cost = c.cheapest_feasible_insertion(u, self.dist_matrix)

            # attempt insertion
            try:
                if opts['verbose'] > 0:
                    print(f'\tInserting {u.id_} into {carrier.id_}.{vehicle.id_} with cost of {round(cost, 2)}')
                vehicle.tour.insert_and_reset_schedules(position, u)
                vehicle.tour.compute_cost_and_schedules(self.dist_matrix)
                carrier.unrouted.pop(u.id_)  # remove inserted request from unrouted
            except TypeError:
                raise InsertionError('', f"Cannot insert {u} feasibly into {carrier.id_}.{vehicle.id_}")

        return

    def to_centralized(self, depot_xy: tuple):
        central_depot = vx.Vertex('d_central', *depot_xy, 0, 0, float('inf'))
        central_vehicles = []
        for c in self.carriers:
            central_vehicles.extend(c.vehicles)
        central_carrier = cr.Carrier(0, central_depot, central_vehicles)
        centralized = Instance(self.id_ + 'centralized', self.requests, [central_carrier])
        for r in centralized.requests:
            centralized.carriers[0].assign_request(r)
        return centralized

    def plot(self, annotate: bool = True, alpha: float = 1):
        for c in self.carriers:
            plt.plot(*c.depot.coords, marker='s', alpha=alpha, linestyle='', c='black')
            r_x_coords = [r.coords.x for r in c.requests.values()]
            r_y_coords = [r.coords.y for r in c.requests.values()]
            plt.plot(r_x_coords, r_y_coords, marker='o', alpha=alpha, label=c.id_, ls='')

        if annotate:
            texts = [plt.text(*c.depot.coords, s=c.depot.id_) for c in self.carriers]
            adjust_text(texts)

        plt.gca().legend()
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.title(f'Instance {self.id_}')

        return

    def write_custom_json(self):
        file_name = f'../data/Custom/{self.id_}'

        # check if any customers are assigned to carriers already and set the file name
        assigned_requests = []
        for c in self.carriers:
            assigned_requests.extend(c.requests)
        if any(assigned_requests):
            file_name += '_ass'

        # check how many instances of this type already have been stored and enumerate file name accordingly
        listdir = os.listdir(f'../data/Custom')
        enum = 0
        for file in listdir:
            if self.id_ in file:
                enum += 1
        file_name += f'_#{enum}'

        file_name += '.json'
        with open(file_name, mode='w') as write_file:
            json.dump(self.to_dict(), write_file, indent=4)
        return file_name


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
    self.assign_all_requests_randomly()
    self.write_custom_json()
    # custom = read_custom_json(f'R101_{num_carriers}_{num_vehicles}')
    pass
