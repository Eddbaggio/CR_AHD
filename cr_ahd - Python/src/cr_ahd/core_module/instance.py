import json
import logging.config
import multiprocessing
from pathlib import Path
from typing import List
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from tqdm import tqdm
from src.cr_ahd.utility_module import istarmap
from scipy.spatial.distance import pdist, squareform

from src.cr_ahd.core_module import carrier as cr, vehicle as vh, vertex as vx, request as rq
import src.cr_ahd.utility_module.utils as ut

logger = logging.getLogger(__name__)


class PDPInstance:
    def __init__(self, id_: str,
                 requests: pd.DataFrame,
                 carrier_depots: pd.DataFrame,
                 carriers_max_num_vehicles: int,
                 vehicles_max_load: float,
                 vehicles_max_tour_length: float):
        """
        Create an instance for the collaborative transportation network for attended home delivery

        :param id_: unique identifier
        """
        self._id_ = id_
        self.vehicles_max_load = vehicles_max_load
        self.vehicles_max_travel_distance = vehicles_max_tour_length
        self.carrier_depots = carrier_depots
        self.carriers_max_num_vehicles = carriers_max_num_vehicles
        self.requests: np.ndarray = requests.index.values
        self.x_coords: np.ndarray = np.concatenate([carrier_depots['x'], requests['pickup_x'], requests['delivery_x']])
        self.y_coords: np.ndarray = np.concatenate([carrier_depots['y'], requests['pickup_y'], requests['delivery_y']])
        self.request_to_carrier_assignment = requests['carrier_index']
        self.revenue = np.concatenate(
            [np.zeros(self.num_carriers), np.zeros_like(requests['revenue']), requests['revenue']])
        self.load = np.concatenate(
            [np.zeros(self.num_carriers), requests['load'], -requests['load']])
        self.service_time = tuple(
            [*pd.Series([pd.Timedelta(0)]*self.num_carriers), *requests['service_time'], *requests['service_time']])

        # compute the distance matrix
        request_coordinates = pd.concat([
            requests[['pickup_x', 'pickup_y']].rename(columns={'pickup_x': 'x', 'pickup_y': 'y'}),
            requests[['delivery_x', 'delivery_y']].rename(columns={'delivery_x': 'x', 'delivery_y': 'y'})],
            keys=['pickup', 'delivery']
        ).swaplevel()
        self._distance_matrix = squareform(pdist(np.concatenate([carrier_depots, request_coordinates]), 'euclidean'))
        self.df = requests  # store the data as pd.DataFrame as well
        logger.debug(f'{id_}: created')

    def __str__(self):
        return f'Instance {self.id_} with {len(self.requests)} customers and {len(self.carrier_depots)} carriers'

    @property
    def id_(self):
        return self._id_

    @property
    def num_carriers(self):
        """The number of carriers in the Instance"""
        return len(self.carrier_depots)

    @property
    def num_requests(self):
        """The number of requests"""
        return len(self.requests)

    def distance(self, i, j):
        return self._distance_matrix[i, j]

    def pickup(self, request: int):
        """returns the pickup vertex index for the given request"""
        return self.num_carriers + request

    def delivery(self, request: int):
        """returns the delivery vertex index for the given request"""
        return self.num_carriers + self.num_requests + request

    def pickup_delivery_pair(self, request: int):
        """returns a tuple of pickup & delivery vertex indices for the given request"""
        return self.pickup(request), self.delivery(request)

    # @property
    # def sum_travel_distance(self):
    #     """The total routing distance over all carriers over all vehicles"""
    #     total_distance = 0
    #     for c in self.carriers:
    #         for v in c.vehicles:
    #             total_distance += v.tour.sum_travel_distance
    #     return total_distance

    '''
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
    def num_requests_per_carrier_per_vehicle(self):
        num_requests_per_carrier_per_vehicle = dict()
        for c in self.carriers:
            for v in c.active_vehicles:
                num_requests_per_carrier_per_vehicle[f'{c.id_}_{v.tour.id_}_num_requests'] = len(v.tour)
        return num_requests_per_carrier_per_vehicle

    @property
    def travel_distance_per_carrier(self):
        travel_distance_per_carrier = dict()
        for c in self.carriers:
            travel_distance_per_carrier[f'{c.id_}_travel_distance'] = c.sum_travel_distance()
        return travel_distance_per_carrier

    @property
    def travel_distance_per_carrier_per_vehicle(self):
        travel_distance_per_carrier_per_vehicle = dict()
        for c in self.carriers:
            for v in c.active_vehicles:
                travel_distance_per_carrier_per_vehicle[
                    f'{c.id_}_{v.tour.id_}_travel_distance'] = v.tour.sum_travel_distance
        return travel_distance_per_carrier_per_vehicle

    @property
    def travel_duration_per_carrier(self):
        travel_duration_per_carrier = dict()
        for c in self.carriers:
            travel_duration_per_carrier[f'{c.id_}_travel_duration'] = c.sum_travel_duration()
        return travel_duration_per_carrier

    @property
    def travel_duration_per_carrier_per_vehicle(self):
        travel_duration_per_carrier_per_vehicle = dict()
        for c in self.carriers:
            for v in c.active_vehicles:
                travel_duration_per_carrier_per_vehicle[
                    f'{c.id_}_{v.tour.id_}_travel_duration'] = v.tour.sum_travel_duration
        return travel_duration_per_carrier_per_vehicle

    @property
    def revenue_per_carrier(self):
        revenue_per_carrier = dict()
        for c in self.carriers:
            revenue_per_carrier[f'{c.id_}_revenue'] = c.revenue
        return revenue_per_carrier

    @property
    def revenue_per_carrier_per_vehicle(self):
        revenue_per_carrier_per_vehicle = dict()
        for c in self.carriers:
            for v in c.active_vehicles:
                revenue_per_carrier_per_vehicle[f'{c.id_}_{v.tour.id_}_revenue'] = v.tour.revenue
        return revenue_per_carrier_per_vehicle

    @property
    def profit_per_carrier(self):
        profit_per_carrier = dict()
        for c in self.carriers:
            profit_per_carrier[f'{c.id_}_profit'] = c.profit
        return profit_per_carrier

    @property
    def profit_per_carrier_per_vehicle(self):
        profit_per_carrier_per_vehicle = dict()
        for c in self.carriers:
            for v in c.active_vehicles:
                profit_per_carrier_per_vehicle[f'{c.id_}_{v.tour.id_}_profit'] = v.tour.profit
        return profit_per_carrier_per_vehicle

    @property
    def solution(self):
        """The instance's current solution as a python dictionary"""
        instance_solution = {}
        for c in self.carriers:
            carrier_solution = {}
            v: vh.Vehicle
            for v in c.active_vehicles:
                vehicle_solution = dict(
                    routing_sequence=[r.id_ for r in v.tour.routing_sequence],
                    distance_sequence=v.tour.travel_distance_sequence,
                    distance_sequence_cumul=np.cumsum(v.tour.travel_distance_sequence).tolist(),
                    duration_sequence=v.tour.travel_duration_sequence,
                    duration_sequence_cumul=np.cumsum(v.tour.travel_duration_sequence).tolist(),
                    arrival_schedule=v.tour.arrival_schedule,
                    service_schedule=v.tour.service_schedule,
                )
                carrier_solution[v.id_] = vehicle_solution
            instance_solution[c.id_] = carrier_solution
        return instance_solution

    @property
    def solution_summary(self):
        return dict(id=self.id_,
                    rand_copy=self.id_.split('#')[-1],
                    solomon_base=self.solomon_base,
                    num_carriers=self.num_carriers,
                    num_requests=self.num_requests,
                    num_vehicles=self.num_vehicles,
                    num_act_veh=self.num_act_veh,
                    travel_distance=self.sum_travel_distance,
                    travel_duration=self.sum_travel_duration,
                    revenue=self.revenue,
                    profit=self.profit,
                    solution_algorithm=self.solution_algorithm.__class__.__name__.replace('Solver', ''),
                    **self.num_act_veh_per_carrier,
                    **self.num_requests_per_carrier,
                    **self.num_requests_per_carrier_per_vehicle,
                    **self.travel_distance_per_carrier,
                    **self.travel_distance_per_carrier_per_vehicle,
                    **self.travel_duration_per_carrier,
                    **self.travel_duration_per_carrier_per_vehicle,
                    **self.revenue_per_carrier,
                    **self.revenue_per_carrier_per_vehicle,
                    **self.profit_per_carrier,
                    **self.profit_per_carrier_per_vehicle,
                    )

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
    '''

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
        centralized = PDPInstance(self.id_, self.requests, [central_carrier])
        centralized.carriers[0].assign_requests(centralized.requests)
        return centralized

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
            json.dump(self.to_dict(), write_file, indent=4, cls=ut.MyJSONEncoder)
        # LOGGER.debug(f'{self.id_}: wrote instance to json at {file_name}')
        return file_name


def read_solomon(name: str, num_carriers: int) -> PDPInstance:
    """
    Read in a specified Solomon instance and add a specified number of carriers to it.

    :param name: file name of the Solomon instance (without file extension)
    :param num_carriers: number of carriers to be added to the instance
    :return: realization of the Instance class
    """
    file_name = ut.path_input_solomon / f'{name}.txt'

    # vehicles
    vehicles = pd.read_csv(file_name, skiprows=4, nrows=1, delim_whitespace=True, names=['number', 'capacity'])
    vehicles = [vh.Vehicle('v' + str(i), vehicles.load_capacity[0], vehicles.load_capacity[0]) for i in
                range(vehicles.number[0])]
    vehicles = ut.split_iterable(vehicles, num_carriers)

    # requests
    cols = ['cust_no', 'x_coord', 'y_coord', 'demand', 'ready_time', 'due_date', 'service_duration']
    requests = pd.read_csv(file_name, skiprows=10, delim_whitespace=True, names=cols)
    requests = [
        rq.DeliveryRequest('r' + str(row.cust_no - 1), row.x_coord, row.y_coord,
                           tw_open=dt.datetime.min + dt.timedelta(minutes=row.ready_time),
                           tw_close=dt.datetime.min + dt.timedelta(minutes=row.due_date),
                           carrier_assignment=None,
                           revenue=0,
                           load=row.demand,
                           )
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
    inst = PDPInstance(name, requests, carriers)
    if ut.opts['verbose'] > 0:
        print(f'Successfully read Solomon {name} with {num_carriers} carriers.')
        print(inst)
    return inst


def read_gansterer_hartl_mv(path: Path) -> PDPInstance:
    """read an instance file as used in (Gansterer,M., & Hartl,R.F. (2016). Request evaluation strategies
    for carriers in auction-based collaborations. https://doi.org/10.1007/s00291-015-0411-1).
    multiplies the max vehicle load by 10!
    """
    vrp_params = pd.read_csv(path, skiprows=1, nrows=3, delim_whitespace=True, header=None, squeeze=True, index_col=0)
    depots = pd.read_csv(path, skiprows=7, nrows=3, delim_whitespace=True, header=None, index_col=False,
                         usecols=[1, 2], names=['x', 'y'])
    cols = ['carrier_index', 'pickup_x', 'pickup_y', 'delivery_x', 'delivery_y', 'revenue', 'load']
    requests = pd.read_csv(path, skiprows=13, delim_whitespace=True, names=cols, index_col=False,
                           float_precision='round_trip')
    requests['service_time'] = dt.timedelta(0)
    return PDPInstance(path.stem, requests, depots, vrp_params['V'], vrp_params['L']*10, vrp_params['T'])


# def read_gansterer_hartl_mv(path: Path) -> Instance:
#     # requests
#     cols = ['carrier_index', 'pickup_x', 'pickup_y', 'delivery_x', 'delivery_y', 'revenue', 'load']
#     requests = pd.read_csv(path, skiprows=13, delim_whitespace=True, names=cols, index_col=False,
#                            float_precision='round_trip')
#     requests = [rq.PickupAndDeliveryRequest(
#         id_=f'r{int(index)}',
#         pickup_x=row.pickup_x, pickup_y=row.pickup_y, delivery_x=row.delivery_x, delivery_y=row.delivery_y,
#         delivery_tw_open=ut.START_TIME, delivery_tw_close=ut.END_TIME,
#         carrier_assignment=f'c{int(row.carrier_index)}', revenue=row.revenue, load=row.load
#     ) for index, row in requests.iterrows()]
#     request_vertices = [*[r.pickup_vertex for r in requests], *[r.delivery_vertex for r in requests]]
#     # depots
#     depots = pd.read_csv(path, skiprows=7, nrows=3, delim_whitespace=True, header=None, index_col=False,
#                          usecols=[1, 2], names=['x', 'y'])
#     depots = [vx.DepotVertex(f'd{index}', row.x, row.y, f'c{index}') for index, row in depots.iterrows()]
#     # global distance matrix
#     distance_matrix = ut.make_travel_dist_matrix([*depots, *request_vertices])
#     # carriers + vehicles
#     vehicles = pd.read_csv(path, skiprows=1, nrows=3, delim_whitespace=True, header=None, squeeze=True, index_col=0)
#     carriers = []
#     for i, depot in enumerate(depots):
#         c_vehicles = [vh.Vehicle(f'v{str(j + vehicles.V * i)}', vehicles.L, vehicles.T) for j in range(vehicles.V)]
#         c_requests = [r for r in requests if r.carrier_assignment == f'c{i}']
#         carriers.append(cr.Carrier(f'c{i}', depot, c_vehicles, c_requests, distance_matrix))
#     return Instance(path.stem, requests, carriers, distance_matrix)


def make_custom_from_solomon(solomon: PDPInstance, custom_name: str, num_carriers: int, num_vehicles_per_carrier: int,
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
    inst = PDPInstance(custom_name, solomon.requests, carriers, solomon.distance_matrix)
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
        request_dict['service_duration'] = dt.timedelta(seconds=request_dict['service_duration'])
        request = vx.Vertex(**request_dict)
        requests.append(request)

    inst = PDPInstance(path.stem, requests, carriers, dist_matrix)
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
    """
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
    """
    p = Path("C:/Users/Elting/ucloud/PhD/02_Research/02_Collaborative Routing for Attended Home Deliveries/01_Code/"
             "data/Input/Gansterer_Hartl/3carriers/MV_instances/run=0+dist=200+rad=150+n=10.dat")
    inst = read_gansterer_hartl_mv(p)
    pass
