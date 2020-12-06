import json
import os
import warnings
from copy import copy
from itertools import islice
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from tqdm import tqdm

import carrier as cr
import vehicle as vh
import vertex as vx
from plotting import CarrierConstructionAnimation
from profiling import Timer
from utils import split_iterable, make_dist_matrix, opts, InsertionError, Solomon_Instances, path_output, path_input_custom, \
    path_input_solomon


class Instance(object):
    def __init__(self, id_: str, requests: List[vx.Vertex], carriers: List[cr.Carrier],
                 dist_matrix: pd.DataFrame = None):
        """

        :param id_:
        :param requests:
        :param carriers:
        :param dist_matrix:
        """
        self.id_ = id_
        self.requests = requests
        self.carriers = carriers
        if dist_matrix is not None:
            self.dist_matrix = dist_matrix
        else:
            self.dist_matrix = make_dist_matrix([*self.requests, *[c.depot for c in self.carriers]])
        self._solved = False

    def __str__(self):
        return f'{"Solved " if self._solved else ""}Instance {self.id_} with {len(self.requests)} customers and {len(self.carriers)} carriers'

    @property
    def solomon_base(self):
        return self.id_.split('_')[0]

    @property
    def num_carriers(self):
        return len(self.carriers)

    @property
    def num_vehicles(self):
        return sum([c.num_vehicles for c in self.carriers])

    @property
    def num_requests(self):
        return len(self.requests)

    @property
    def cost(self):
        total_cost = 0
        for c in self.carriers:
            for v in c.vehicles:
                total_cost += v.tour.cost
        return total_cost

    @property
    def revenue(self):
        return sum([c.revenue for c in self.carriers])

    @property
    def profit(self):
        return self.revenue - self.cost

    @property
    def num_act_veh(self):
        num_vehicles_in_use = 0
        for c in self.carriers:
            for v in c.vehicles:
                if len(v.tour) > 2:
                    num_vehicles_in_use += 1
        return num_vehicles_in_use

    @property
    def solution(self):
        instance_solution = {}
        for c in self.carriers:
            carrier_solution = {}
            for v in c.vehicles:
                vehicle_solution = dict(sequence=[r.id_ for r in v.tour.sequence],
                                        arrival=v.tour.arrival_schedule,
                                        service=v.tour.service_schedule,
                                        cost=v.tour.cost)
                carrier_solution[v.id_] = vehicle_solution
                carrier_solution['cost'] = c.cost()
            instance_solution[c.id_] = carrier_solution
            instance_solution['cost'] = self.cost
        return instance_solution

    @property
    def num_act_veh_per_carrier(self):
        num_act_veh_per_carrier = dict()
        for c in self.carriers:
            num_act_veh_per_carrier[f'{c.id_}_num_act_veh'] = c.num_act_veh
        return num_act_veh_per_carrier

    @property
    def num_requests_per_carrier(self):
        num_requests_per_carrier = dict()
        for c in self.carriers:
            num_requests_per_carrier[f'{c.id_}_num_requests'] = len(c.requests)
        return num_requests_per_carrier

    @property
    def cost_per_carrier(self):
        cost_per_carrier = dict()
        for c in self.carriers:
            cost_per_carrier[f'{c.id_}_cost'] = c.cost()
        return cost_per_carrier

    @property
    def revenue_per_carrier(self):
        revenue_per_carrier = dict()
        for c in self.carriers:
            revenue_per_carrier[f'{c.id_}_revenue'] = c.revenue
        return revenue_per_carrier

    @property
    def profit_per_carrier(self):
        profit_per_carrier = dict()
        for c in self.carriers:
            profit_per_carrier[f'{c.id_}_profit'] = c.profit
        return profit_per_carrier

    @property
    def evaluation_metrics(self):
        assert self._solved, f'{self} has not been solved yet'
        return dict(
            id=self.id_,
            rand_copy=self.id_.split('#')[-1],
            solomon_base=self.solomon_base,
            num_carriers=self.num_carriers,
            num_requests=self.num_requests,
            num_vehicles=self.num_vehicles,
            num_act_veh=self.num_act_veh,
            cost=self.cost,
            revenue=self.revenue,
            profit=self.profit,
            runtime=self._runtime,
            algorithm=self._solution_algorithm,
            **self.num_act_veh_per_carrier,
            **self.num_requests_per_carrier,
            **self.cost_per_carrier,
            **self.revenue_per_carrier,
            **self.profit_per_carrier,
        )

    def to_dict(self):
        return {
            'id_': self.id_,
            'requests': [r.to_dict() for r in self.requests],
            'carriers': [c.to_dict() for c in self.carriers],
            'dist_matrix': self.dist_matrix.to_dict()
        }

    def _assign_all_requests_randomly(self):
        """
        Only call this method for creating new instances. If the instance has been read from disk, the assignment is
        stored in there already and should be retained to ensure comparability between methods, even dynamic methods.
                """
        raise UserWarning('Only call this method for constructing new instances & writing them to disk.')
        assert not self._solved, f'Instance {self} has already been solved'
        # np.random.seed(0)
        for r in self.requests:
            c = self.carriers[np.random.choice(range(len(self.carriers)))]
            c.assign_request(r)

    def static_CI_construction(self,
                               verbose: int = opts['verbose'],
                               plot_level: int = opts['plot_level']):
        """
        Use a static construction method to build tours for all carriers via SEQUENTIAL CHEAPEST INSERTION.
        (Can also be used for dynamic route construction if the request-to-carrier assignment is known.)

        :param verbose:
        :param plot_level:
        :return:
        """
        assert not self._solved, f'Instance {self} has already been solved'
        if verbose > 0:
            print(f'STATIC Cheapest Insertion Construction for {self}:')
        timer = Timer()
        timer.start()

        for c in self.carriers:
            if plot_level > 1:
                ani = CarrierConstructionAnimation(c, f'{self.id_}{" centralized" if self.num_carriers == 0 else ""}: '
                                                      f'Cheapest Insertion construction: {c.id_}')

            # TODO why is this necessary here? why aren't these things computed already before?
            c.compute_all_vehicle_cost_and_schedules(self.dist_matrix)

            # TODO no initialization of vehicle tours is done while I1 has initialization - is it unfair?

            # construction loop
            for _ in range(len(c.unrouted)):
                key, u = c.unrouted.popitem(last=False)  # sequential removal from list of unrouted from first to last
                vehicle, position, _ = c.find_cheapest_feasible_insertion(u, self.dist_matrix)

                if verbose > 0:
                    print(f'\tInserting {u.id_} into {c.id_}.{vehicle.tour.id_}')
                vehicle.tour.insert_and_reset(index=position, vertex=u)
                vehicle.tour.compute_cost_and_schedules(self.dist_matrix)

                if plot_level > 1:
                    ani.add_current_frame()

            assert len(c.unrouted) == 0  # just to be completely sure

            if plot_level > 1:
                ani.show()
                file_name = f'{self.id_}_{"cen_" if self.num_carriers == 1 else ""}sta_CI_{c.id_ if self.num_carriers > 1 else ""}.gif'
                ani.save(filename=path_output.joinpath('Animations', file_name))
            if verbose > 0:
                print(f'Total Route cost of carrier {c.id_}: {c.cost()}\n')

        timer.stop()
        self._runtime = timer.duration
        self._solved = True
        self._solution_algorithm = f'{"cen_" if self.num_carriers == 1 else ""}sta_CI'
        return

    def static_I1_construction(self,
                               init_method: str = 'earliest_due_date',
                               verbose: int = opts['verbose'],
                               plot_level: int = opts['plot_level']):
        """
        Use the I1 construction method (Solomon 1987) to build tours for all carriers.

        :param init_method: pendulum tours are created if necessary. 'earliest_due_date' or 'furthest_distance' methods are available
        :param verbose:
        :param plot_level:
        :return:
        """
        assert not self._solved, f'Instance {self} has already been solved'
        if verbose > 0:
            print(f'STATIC I1 Construction for {self}:')
        timer = Timer()
        timer.start()

        for c in self.carriers:
            if plot_level > 1:
                ani = CarrierConstructionAnimation(c,
                                                   f"{self.id_}{' centralized' if self.num_carriers == 0 else ''}: Solomon's I1 construction: {c.id_}")

            # TODO why is this necessary here? why aren't these things computed already before?
            c.compute_all_vehicle_cost_and_schedules(self.dist_matrix)  # tour empty at this point (depot to depot tour)

            # initialize one tour to begin with
            c.initialize_tour(c.inactive_vehicles[0], self.dist_matrix, init_method)

            # construction loop
            while any(c.unrouted):
                u, vehicle, position, _ = c.find_best_feasible_I1_insertion(self.dist_matrix)
                if position is not None:  # insert
                    c.unrouted.pop(u.id_)
                    vehicle.tour.insert_and_reset(index=position, vertex=u)
                    vehicle.tour.compute_cost_and_schedules(self.dist_matrix)
                    if verbose > 0:
                        print(f'\tInserting {u.id_} into {c.id_}.{vehicle.tour.id_}')
                    if plot_level > 1:
                        ani.add_current_frame()
                else:
                    if any(c.inactive_vehicles):
                        c.initialize_tour(c.inactive_vehicles[0], self.dist_matrix, init_method)
                        if plot_level > 1:
                            ani.add_current_frame()
                    else:
                        InsertionError('', 'No more vehicles available')

            assert len(c.unrouted) == 0  # just to be on the safe side

            if plot_level > 1:
                ani.show()
                ani.save(
                    filename=f'.{path_output}/Animations/{self.id_}_{"cen_" if self.num_carriers == 1 else ""}sta_I1_{c.id_ if self.num_carriers > 1 else ""}.gif')
            if verbose > 0:
                print(f'Total Route cost of carrier {c.id_}: {c.cost()}\n')

        timer.stop()
        self._runtime = timer.duration
        self._solved = True
        self._solution_algorithm = f'{"cen_" if self.num_carriers == 1 else ""}sta_I1'
        return

    def cheapest_insertion_auction(self, request: vx.Vertex, initial_carrier: cr.Carrier, verbose=opts['verbose'], ):
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

    def dynamic_construction(self, with_auction: bool = True, verbose=opts['verbose'], plot_level=opts['plot_level']):
        assert not self._solved, f'Instance {self} has already been solved'

        if verbose > 0:
            print(
                f'DYNAMIC Cheapest Insertion Construction {"WITH" if with_auction else "WITHOUT"} auction for {self}:')

        timer = Timer()
        timer.start()
        # find the next request u, that has id number i
        # TODO this can be simplified/made more efficient if the assignment of a vertex is stored with its class
        #  instance. In that case, it must also be stored accordingly in the json file. Right now, it is not a big
        #  problem since requests are assigned in ascending order, so only the first request of each carrier must be
        #  checked

        # TODO this function assumes that requests have been assigned to carriers already which is not really logical
        #  in a real-life case since they arrive dynamically
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
                    c.retract_request(u.id_)
                    carrier.assign_request(u)  # assign to auction winner
            else:
                # find cheapest insertion
                carrier = c
                vehicle, position, cost = c.find_cheapest_feasible_insertion(u, self.dist_matrix)

            # attempt insertion
            try:
                if opts['verbose'] > 0:
                    print(f'\tInserting {u.id_} into {carrier.id_}.{vehicle.id_} with cost of {round(cost, 2)}')
                vehicle.tour.insert_and_reset(position, u)
                vehicle.tour.compute_cost_and_schedules(self.dist_matrix)
                carrier.unrouted.pop(u.id_)  # remove inserted request from unrouted
            except TypeError:
                raise InsertionError('', f"Cannot insert {u} feasibly into {carrier.id_}.{vehicle.id_}")

        timer.stop()
        self._runtime = timer.duration
        self._solved = True
        self._solution_algorithm = f'dyn{"_auc" if with_auction else ""}'
        return

    def static_auction(self):
        for c in self.carriers:
            c.determine_auction_set()

    def two_opt(self):
        # print(f'Before 2-opt: {self.cost_per_carrier}')
        for c in self.carriers:
            c.two_opt(self.dist_matrix)
        # print(f'After 2-opt: {self.cost_per_carrier}')
        self._solution_algorithm += '_2opt'

    def to_centralized(self, depot_xy: tuple):
        central_depot = vx.Vertex('d_central', *depot_xy, 0, 0, float('inf'))
        central_vehicles = []
        for c in self.carriers:
            central_vehicles.extend(c.vehicles)
        central_carrier = cr.Carrier('c0', central_depot, central_vehicles)
        centralized = Instance(self.id_, self.requests, [central_carrier])
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
        plt.grid()
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.title(f'Instance {self.id_}')

        return

    def write_instance_to_json(self):
        file_name = f'{path_input_custom}/{self.solomon_base}/{self.id_}'
        os.makedirs(f'{path_input_custom}/{self.solomon_base}', exist_ok=True)

        # check if any customers are assigned to carriers already and set the file name
        assigned_requests = []
        for c in self.carriers:
            assigned_requests.extend(c.requests)
        if any(assigned_requests):
            file_name += '_ass'

        # check how many instances of this type already have been stored and enumerate file name accordingly
        listdir = os.listdir(f'{path_input_custom}/{self.solomon_base}')
        enum = 0
        for file in listdir:
            if self.id_ in file:
                enum += 1
        file_name += f'_#{enum:03d}'

        file_name += '.json'
        with open(file_name, mode='w') as write_file:
            json.dump(self.to_dict(), write_file, indent=4)
        return file_name

    def write_solution_to_json(self):
        assert self._solved
        dir_name = f'{path_input_custom}/{self.solomon_base}/'  # TODO why do I save this in the input folder?
        file_name = f'{self.id_}_{self._solution_algorithm}_sol.json'
        path = os.path.join(dir_name, file_name)
        os.makedirs(dir_name, exist_ok=True)
        if os.path.exists(path):
            # raise FileExistsError
            warnings.warn('Existing solution file is overwritten')
        with open(path, mode='w') as write_file:
            json.dump(self.solution, write_file, indent=4)
        return path

    # def write_evaluation_metrics_to_csv(self):
    #     assert self._solved
    #     file_name = f'../data/Output/Custom/{self.solomon_base}/{self.id_}_{self._solution_algorithm}_eval.csv'
    #     df = pd.Series(self.evaluation_metrics)
    #     df.to_csv(file_name)
    #     return file_name


def read_solomon(name: str, num_carriers: int) -> Instance:
    # TODO: a more efficient way of reading the data, e.g. by reading all vertices at once and split off the depot
    #  after; and by not using pd to read vehicles
    vehicles = pd.read_csv(f'{path_input_solomon}/{name}.txt', skiprows=4, nrows=1, delim_whitespace=True,
                           names=['number', 'capacity'])
    vehicles = [vh.Vehicle('v' + str(i), vehicles.capacity[0]) for i in range(vehicles.number[0])]
    vehicles = split_iterable(vehicles, num_carriers)

    cols = ['cust_no', 'x_coord', 'y_coord', 'demand', 'ready_time', 'due_date', 'service_duration']
    requests = pd.read_csv(f'{path_input_solomon}/{name}.txt', skiprows=10, delim_whitespace=True, names=cols)
    requests = [
        vx.Vertex('r' + str(row.cust_no - 1), row.x_coord, row.y_coord, row.demand, row.ready_time, row.due_date) for
        row in requests.itertuples()]

    depot = pd.read_csv(f'{path_input_solomon}/{name}.txt', skiprows=9, nrows=1, delim_whitespace=True,
                        names=cols)  # TODO: how to handle depots? read in read_solomon?
    depot = vx.Vertex('d1', int(depot.x_coord[0]), int(depot.y_coord[0]), int(depot.demand[0]),
                      int(depot.ready_time[0]),
                      int(depot.due_date[0]))
    carriers = []
    for i in range(num_carriers):
        c = cr.Carrier(f'c{i}', copy(depot), next(vehicles))
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
        carrier = cr.Carrier(f'c{i}',
                             depot=solomon.carriers[i].depot,
                             vehicles=vehicles)
        carriers.append(carrier)
    inst = Instance(custom_name, solomon.requests, carriers)
    if verbose > 0:
        print(f'Created custom instance {inst}')
    return inst


def read_custom_json_instance(path: str):
    # file_name = f'../data/Custom/{name}.json'
    with open(path, 'r') as reader_file:
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
    _, name = os.path.split(path)
    name = name.split(sep='.')[0]
    inst = Instance(name, requests, carriers, dist_matrix)
    return inst


if __name__ == '__main__':
    # If you need to re-create the custom instances run the following chunk:
    # num_carriers = 3
    # num_vehicles = 15
    # for solomon in tqdm(Solomon_Instances):
    #     for i in range(100):
    #         # TODO read solomon only once
    #         self = make_custom_from_solomon(solomon,
    #                                         f'{solomon}_{num_carriers}_{num_vehicles}',
    #                                         num_carriers,
    #                                         num_vehicles,
    #                                         None)
    #         self._assign_all_requests_randomly()
    #         self.write_instance_to_json()
    pass
