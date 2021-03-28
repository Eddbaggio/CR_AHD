import json
import logging.config
import multiprocessing
import time
from pathlib import Path
from typing import List
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from tqdm import tqdm
from src.cr_ahd.utility_module import istarmap
from src.cr_ahd.utility_module.utils import DateTimeEncoder

import src.cr_ahd.core_module.carrier as cr
import src.cr_ahd.core_module.vehicle as vh
import src.cr_ahd.core_module.vertex as vx
import src.cr_ahd.utility_module.utils as ut
from src.cr_ahd.core_module.optimizable import Optimizable

logger = logging.getLogger(__name__)


class Instance(Optimizable):
    def __init__(self, id_: str, requests: List[vx.Vertex], carriers: List[cr.Carrier],
                 dist_matrix: pd.DataFrame = None):
        """
        Create an instance (requests, carriers (,distance matrix) for the collaborative transportation network for
        attended home delivery

        :param id_: unique identifier (commonly contains info about solomon base instance, number of carriers and number
         etc.
        :param requests: the requests that belong to the instance, i.e. all Vertices that mus be visited (excluding
        depots) in a list of type vx.Vertex elements
        :param carriers: the participating carriers (of class cr.Carrier) of this instance's transportation network
        :param dist_matrix: The distance matrix specifying distances between all request- and depot-vertices
        """
        self.id_ = id_
        self.requests = requests
        self.carriers = carriers
        if dist_matrix is not None:
            self._distance_matrix = dist_matrix
        else:
            self._distance_matrix = ut.make_travel_dist_matrix([*self.requests, *[c.depot for c in self.carriers]])
        self.solution_algorithm = None
        # LOGGER.debug(f'{self.id_}: created')
        pass

    def __str__(self):
        return f'Instance {self.id_} with {len(self.requests)} customers and {len(self.carriers)} carriers'

    @property
    def distance_matrix(self):
        return self._distance_matrix

    @distance_matrix.setter
    def distance_matrix(self, dist_matrix):
        self._distance_matrix = dist_matrix

    @property
    def solomon_base(self):
        """The Solomon Instance's name on which self is based on"""
        return self.id_.split('_')[0]

    @property
    def num_carriers(self):
        """The number of carriers in the Instance"""
        return len(self.carriers)

    @property
    def num_vehicles(self):
        """The TOTAL  number of vehicles in the instance"""
        return sum([c.num_vehicles for c in self.carriers])

    @property
    def num_requests(self):
        """The number of requests"""
        return len(self.requests)
        pass

    @property
    def sum_travel_distance(self):
        """The total routing distance over all carriers over all vehicles"""
        total_distance = 0
        for c in self.carriers:
            for v in c.vehicles:
                total_distance += v.tour.sum_travel_distance
        return total_distance

    @property
    def sum_travel_duration(self):
        """The total routing duration over all carriers over all vehicles"""
        total_duration = dt.timedelta()
        for c in self.carriers:
            for v in c.vehicles:
                total_duration += v.tour.sum_travel_duration
        return total_duration

    @property
    def revenue(self):
        return sum([c.revenue for c in self.carriers])

    @property
    def profit(self):
        return self.revenue - self.sum_travel_distance

    @property
    def num_act_veh(self):
        """The total number of active vehicles over all carriers"""
        num_vehicles_in_use = 0
        for c in self.carriers:
            for v in c.vehicles:
                if len(v.tour) > 2:
                    num_vehicles_in_use += 1
        return num_vehicles_in_use

    @property
    def solution(self):
        """The instance's current solution as a python dictionary"""
        instance_solution = {}
        for c in self.carriers:
            carrier_solution = {'travel_distance': c.sum_travel_distance(),
                                'travel_duration': c.sum_travel_duration()}
            v: vh.Vehicle
            for v in c.vehicles:
                vehicle_solution = dict(sequence=[r.id_ for r in v.tour.routing_sequence],
                                        arrival=v.tour.arrival_schedule,
                                        service=v.tour.service_schedule,
                                        travel_distance=v.tour.sum_travel_distance,
                                        travel_duration=v.tour.sum_travel_duration,
                                        )
                carrier_solution[v.id_] = vehicle_solution

            instance_solution[c.id_] = carrier_solution
        instance_solution['travel_duration'] = self.sum_travel_duration
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
    def travel_distance_per_carrier(self):
        travel_distance_per_carrier = dict()
        for c in self.carriers:
            travel_distance_per_carrier[f'{c.id_}_travel_distance'] = c.sum_travel_distance()
        return travel_distance_per_carrier

    @property
    def travel_duration_per_carrier(self):
        travel_duration_per_carrier = dict()
        for c in self.carriers:
            travel_duration_per_carrier[f'{c.id_}_travel_duration'] = c.sum_travel_duration()
        return travel_duration_per_carrier

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
        # assert self.solved, f'{self} has not been solved yet'
        return dict(id=self.id_,
                    rand_copy=self.id_.split('#')[-1],
                    solomon_base=self.solomon_base,
                    num_carriers=self.num_carriers,
                    num_requests=self.num_requests,
                    num_vehicles=self.num_vehicles,
                    num_act_veh=self.num_act_veh,
                    distance=self.sum_travel_distance,
                    duration=self.sum_travel_duration,
                    revenue=self.revenue,
                    profit=self.profit,
                    # initializing_visitor=self.initializing_visitor.__class__.__name__,
                    # initialized=self._initialized,
                    # routing_visitor=self._routing_visitor.__class__.__name__,
                    # solved=self.solved,
                    # finalizing_visitor=self._finalizing_visitor.__class__.__name__,
                    # finalized=self._finalized,
                    solution_algorithm=self.solution_algorithm.__class__.__name__.replace('Solver', ''),
                    # runtime=self._runtime,
                    **self.num_act_veh_per_carrier,
                    **self.num_requests_per_carrier,
                    **self.travel_duration_per_carrier,
                    **self.revenue_per_carrier,
                    **self.profit_per_carrier, )

    @property
    def unrouted_requests(self):
        unrouted = [r for r in self.requests if not r.routed]
        return unrouted

    @property
    def routed_requests(self):
        routed = [r for r in self.requests if r.routed]
        return routed

    def assigned_requests(self):
        assigned_requests = []
        for carrier in self.carriers:
            assigned_requests.extend(carrier.requests)
        return sorted(assigned_requests, key=lambda request: int(request.id_[1:]))

    def unassigned_requests(self):
        unassigned_requests = self.requests[:]
        for assigned_request in self.assigned_requests():
            unassigned_requests.remove(assigned_request)
        return unassigned_requests

    def assigned_unrouted_requests(self):
        return [r for r in self.assigned_requests() if not r.routed]

    def to_dict(self):
        """Convert the instance to a nested dictionary. Primarily useful for storing in .json format"""
        return {
            'id_': self.id_,
            'requests': [r.to_dict() for r in self.requests],
            'carriers': [c.to_dict() for c in self.carriers],
            # 'initialization_visitor': self.initializing_visitor.__class__.__name__,
            # 'routing_visitor': self.routing_visitor.__class__.__name__,
            # 'finalizing_visitor': self.finalizing_visitor.__class__.__name__,
            'dist_matrix': self._distance_matrix.to_dict()
        }

    def retract_all_vertices_from_carriers(self):
        for c in self.carriers:
            c.retract_requests_and_update_routes(c.requests)
        # LOGGER.debug(f'{self.id_}: retracted all vertices from carriers')
        pass

    def _assign_all_requests_randomly(self, random_seed=None):
        """
        Only call this method for creating new instances. If the instance has been read from disk, the assignment is
        stored in there already and should be retained to ensure comparability between methods, even dynamic methods.
        """

        # raise UserWarning('Only call this method for constructing new instances & writing them to disk.')
        # assert not self.solved, f'Instance {self} has already been solved'
        if random_seed:
            np.random.seed(random_seed)

        # assign all vertices
        for r in self.requests:
            c = self.carriers[np.random.choice(range(len(self.carriers)))]
            c.assign_requests([r])
        # LOGGER.debug(f'{self.id_}: assigned all requests randomly')
        pass

    def to_centralized(self, depot_xy: tuple):
        """
        Convert the instance to a centralized instance, i.e. this function returns the same instance but with only a
        single carrier whose depot is at the given depot_xy coordinates

        :param depot_xy: coordinates of the central carrier's depot
        :return: the same instance with just a single carrier
        """
        central_depot = vx.DepotVertex('d_central', *depot_xy, carrier_assignment='d0')
        central_vehicles = []
        for c in self.carriers:
            central_vehicles.extend(c.vehicles)
        central_carrier = cr.Carrier('c0', central_depot, central_vehicles)
        centralized = Instance(self.id_, self.requests, [central_carrier])
        centralized.carriers[0].assign_requests(centralized.requests)
        return centralized

    def plot(self, annotate: bool = True, alpha: float = 1):
        """
        Create a matplotlib plot of the instance, showing the carriers's depots and requests

        :param annotate: whether depots are annotated with their id
        :param alpha:
        :return:
        """
        for c in self.carriers:
            plt.plot(*c.depot.coords, marker='s', alpha=alpha, linestyle='', c='black')  # depots
            r_x_coords = [r.coords.x for r in c.requests]
            r_y_coords = [r.coords.y for r in c.requests]
            plt.plot(r_x_coords, r_y_coords, marker='o', alpha=alpha, label=c.id_, ls='')  # requests

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
        """
        Write the instance's data in a json format

        :return: the file path as a pathlib.Path object
        """
        file_name = ut.path_input_custom.joinpath(self.solomon_base, self.id_)
        # check if any customers are assigned to carriers already and set the file name
        if self.assigned_requests():
            file_name = file_name.with_name(file_name.stem + '_ass')
        file_name = ut.unique_path(file_name.parent, file_name.stem + '_#{:03d}' + '.json')
        file_name.parent.mkdir(parents=True, exist_ok=True)
        with open(file_name, mode='w') as write_file:
            json.dump(self.to_dict(), write_file, indent=4, cls=DateTimeEncoder)
        # LOGGER.debug(f'{self.id_}: wrote instance to json at {file_name}')
        return file_name

    def write_solution_to_json(self):
        """
        Write the instance's solution to a json file
        :return: the file path as a pathlib.Path object
        """
        # assert self.solved
        file_name = f'{self.id_}_{self.solution_algorithm.__class__.__name__}_solution'
        file_path = ut.path_output_custom.joinpath(self.solomon_base, file_name)
        file_path = file_path.with_suffix('.json')
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.exists():
            # raise FileExistsError
            # warnings.warn(f'Existing solution file at {file_path} is overwritten')
            None
        with open(file_path, mode='w') as write_file:
            json.dump(self.solution, write_file, indent=4, cls=DateTimeEncoder)
        # LOGGER.debug(f'{self.id_}: wrote solution to {file_path}')
        return file_path

    # def write_evaluation_metrics_to_csv(self):
    #     assert self.solved
    #     file_name =
    #     f'../data/Output/Custom/{self.solomon_base}/{self.id_}_{self.routing_strategy}_eval.csv'
    #     df = pd.Series(self.evaluation_metrics)
    #     df.to_csv(file_name)
    #     return file_name


def read_solomon(name: str, num_carriers: int) -> Instance:
    """
    Read in a specified Solomon instance and add a specified number of carriers to it.

    :param name: file name of the Solomon instance (without file extension)
    :param num_carriers: number of carriers to be added to the instance
    :return: realization of the Instance class
    """
    file_name = ut.path_input_solomon / f'{name}.txt'

    # vehicles
    vehicles = pd.read_csv(file_name, skiprows=4, nrows=1, delim_whitespace=True, names=['number', 'capacity'])
    vehicles = [vh.Vehicle('v' + str(i), vehicles.capacity[0]) for i in range(vehicles.number[0])]
    vehicles = ut.split_iterable(vehicles, num_carriers)

    # requests
    cols = ['cust_no', 'x_coord', 'y_coord', 'demand', 'ready_time', 'due_date', 'service_duration']
    requests = pd.read_csv(file_name, skiprows=10, delim_whitespace=True, names=cols)
    requests = [
        vx.Vertex('r' + str(row.cust_no - 1), row.x_coord, row.y_coord, row.demand,
                  dt.datetime.min + dt.timedelta(minutes=row.ready_time),
                  dt.datetime.min + dt.timedelta(minutes=row.due_date))
        for row in requests.itertuples()]

    # depots
    depot_df = pd.read_csv(file_name, skiprows=9, nrows=1, delim_whitespace=True, names=cols)
    depots = [vx.DepotVertex(f'd{i}', int(depot_df.x_coord[0]), int(depot_df.y_coord[0])) for i in range(num_carriers)]

    distance_matrix = ut.make_travel_dist_matrix([*requests, *depots])

    # carriers
    carriers = []
    for i in range(num_carriers):
        c = cr.Carrier(f'c{i}', depots[i], next(vehicles), dist_matrix=distance_matrix)
        carriers.append(c)
    inst = Instance(name, requests, carriers)
    if ut.opts['verbose'] > 0:
        print(f'Successfully read Solomon {name} with {num_carriers} carriers.')
        print(inst)
    return inst


def make_custom_from_solomon(solomon: Instance, custom_name: str, num_carriers: int, num_vehicles_per_carrier: int,
                             vehicle_capacity: int, verbose: int = ut.opts['verbose']):
    """
    Create a customized instance for the collaborative transportation network from a Solomon instance with the given
    number of carriers, then adds the defined number of vehicles

    :param solomon: The solomon instance (already read in) that serves as the base for the custom instance
    :param custom_name: name/id for the customized instance
    :param num_carriers: number of carriers in the new custom instance
    :param num_vehicles_per_carrier: number of vehicles that each carrier has at his/her disposal
    :param vehicle_capacity: the loading capacity of each vehicle (08/12/20: afaik currently not in use)
    :param verbose: level of console output
    :return: the custom instance as a realization of the Instance class
    """
    carriers = []  # collect carriers.  these will overwrite the ones created by the read_solomon() function
    for i in range(num_carriers):
        vehicles = []
        for j in range(num_vehicles_per_carrier):
            v = vh.Vehicle('v' + str(i * num_vehicles_per_carrier + j), vehicle_capacity)
            vehicles.append(v)
        depot = solomon.carriers[i].depot
        depot = vx.DepotVertex(depot.id_, depot.coords.x, depot.coords.y, depot.carrier_assignment)
        carrier = cr.Carrier(f'c{i}',
                             depot=depot,
                             vehicles=vehicles,
                             dist_matrix=solomon.distance_matrix)
        carriers.append(carrier)
    inst = Instance(custom_name, solomon.requests, carriers, solomon.distance_matrix)
    if verbose > 0:
        print(f'Created custom instance {inst}')
    return inst


def read_custom_json_instance(path: Path):
    """
    Reading in a custom json instance for the collaborative transportation network. File must have the following structure:
    {
        "id_": "C101_3_15",
        "requests":
            [
                {
                    "id_": "r0"
                    "x_coord": 45,
                    "y_coord": 68,
                    "demand": 10,
                    "tw_open": 912,
                    "tw_close": 967,
                    "service_duration": 0
                },
                {
                    ...
                }
            ],
        "carriers":
            [
                {
                    "id_": "c0",
                    "depot":
                        {
                            "id_": "d0",
                            "x_coord": 40,
                            "y_coord": 50,
                            "demand": 0,
                            "tw_open": 0,
                            "tw_close": 1236,
                            "service_duration": 0
                        },
                    "vehicles":
                        [
                            {
                                "id_": "v0",
                                "capacity": null
                            },
                            {
                                ...
                            },
                        ]
                },
                {
                    "id_": "c1",
                    ...
                },
            ]
        "dist_matrix":
            {
                "r0":
                    {
                        "r0": 0.0,
                        "r1": 2.0,
                        "r2": 3.605551275463989,
                        "r3": 3.0,
                        ...
                    },
                "r1":
                    {
                        ...
                    },
            }
    }

    :param path: Path to the instance
    :return: Instance (requests, carriers, request-to-carrier assignments, ...) for collaborative transportation network
    """

    with open(path, 'r') as reader_file:
        json_data = json.load(reader_file)
    carriers = []
    dist_matrix = pd.DataFrame.from_dict(json_data['dist_matrix'])
    for carrier_dict in json_data['carriers']:
        depot = vx.DepotVertex(**carrier_dict['depot'])
        vehicles = [vh.Vehicle(**v_dict) for v_dict in carrier_dict['vehicles']]
        # assigned_requests = {r_dict['id_']: vx.Vertex(**r_dict) for r_dict in carrier_dict['requests']}
        c = cr.Carrier(carrier_dict['id_'], depot, vehicles, dist_matrix=dist_matrix)
        carriers.append(c)

    requests = []
    request_dict: dict
    for request_dict in json_data['requests']:
        request_dict['tw_open'] = dt.datetime.fromisoformat(request_dict['tw_open'])
        request_dict['tw_close'] = dt.datetime.fromisoformat(request_dict['tw_close'])
        request_dict['service_duration'] = dt.datetime.fromisoformat(request_dict['service_duration']) - dt.datetime.min
        request = vx.Vertex(**request_dict)
        requests.append(request)

    inst = Instance(path.stem, requests, carriers, dist_matrix)
    return inst


def make_and_write_custom_instances(solomon_name, num_carriers, num_vehicles, num_new_instances=100):
    """takes a solomon instance and creates custom instances based on random assignment of requests to carriers.
    Finally, writes the new, custom instances to disk"""

    # LOGGER.info(f'Creating and writing {num_new_instances} new, random instances')
    solomon_instance = read_solomon(solomon_name, num_carriers)
    custom = make_custom_from_solomon(solomon_instance,
                                      # Extract function from this loop, unnecessary to do it every iteration
                                      f'{solomon_name}_{num_carriers}_{num_vehicles}',
                                      num_carriers,
                                      num_vehicles,
                                      None)
    for i in range(num_new_instances):
        custom.retract_all_vertices_from_carriers()
        custom._assign_all_requests_randomly()
        custom.write_instance_to_json()
    pass


def make_and_write_custom_instances_parallel(num_carriers, num_vehicles, num_new_instances,
                                             include_central_instances: bool = True):
    """takes a solomon instance and creates custom instances based on random assignment of requests to carriers.
    Finally, writes the new, custom instances to disk. Does all this with multiprocessing"""
    ut.ask_for_overwrite_permission(ut.path_input_custom)
    iterables = []
    for s in ut.Solomon_Instances:
        for c in [num_carriers]:
            for v in [num_vehicles]:
                for n in [num_new_instances]:
                    iterables.append((s, c, v, n))
        if include_central_instances:
            iterables.append((s, 1, num_carriers * num_vehicles, 1))
    with multiprocessing.Pool() as pool:
        for _ in tqdm(pool.istarmap(make_and_write_custom_instances, iterables), total=len(iterables)):
            pass
    pass


if __name__ == '__main__':
    # If you need to re-create the custom instances run the following:
    start_time = time.time()
    # make_and_write_custom_instances('C101', 3, 15, 3)  # single, for testing
    make_and_write_custom_instances_parallel(num_carriers=3,
                                             num_vehicles=15,
                                             num_new_instances=100,
                                             include_central_instances=True)  # all
    duration = time.time() - start_time
    print(f"Duration: {duration} seconds")
    pass
