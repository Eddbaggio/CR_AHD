import datetime as dt
import json
import logging.config
from pathlib import Path
from typing import Tuple, Sequence, List, NoReturn

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

import utility_module.utils as ut
from tw_management_module import tw

logger = logging.getLogger(__name__)

"""
class Instance(ABC):
    def __init__(self,
                 id_: str,
                 max_num_tours: int,
                 max_vehicle_load: float,
                 max_tour_distance: float,
                 max_tour_duration: dt.timedelta,
                 requests: List[int],
                 requests_initial_carrier_assignment: List[int],
                 requests_disclosure_time: List[dt.datetime],

                 # requests_x: List[float],
                 # requests_y: List[float],
                 # requests_revenue: List[float],
                 # requests_service_duration: List[dt.timedelta],
                 # requests_load: List[float],
                 # request_time_window_open: List[dt.datetime],
                 # request_time_window_close: List[dt.datetime],

                 carrier_depots_x: List[float],
                 carrier_depots_y: List[float],
                 carrier_depots_tw_open: List[dt.datetime],
                 carrier_depots_tw_close: List[dt.datetime],
                 duration_matrix,
                 distance_matrix,
                 ):
        self._id_ = id_
        self.meta = self._meta()
        self.num_carriers = len(carrier_depots_x)
        self.max_vehicle_load = max_vehicle_load
        self.max_tour_distance = max_tour_distance
        self.max_tour_duration = max_tour_duration
        self.max_num_tours = max_num_tours
        self.requests = requests
        self.num_requests = len(self.requests)
        assert self.num_requests % self.num_carriers == 0
        self.num_requests_per_carrier = self.num_requests // self.num_carriers



        self.sanity_check()

    @abstractmethod
    def _meta(self):
        pass

    @abstractmethod
    def sanity_check(self):
        pass

    @abstractmethod
    def calc_duration_matrix(self):
        pass

    @abstractmethod
    def calc_distance_matrix(self):
        pass

    @property
    def id_(self):
        return self._id_
"""


class MDVRPTWInstance:
    def __init__(self,
                 id_: str,
                 carriers_max_num_tours: int,
                 max_vehicle_load: float,
                 max_tour_length: float,
                 max_tour_duration: dt.timedelta,
                 requests: List[int],
                 requests_initial_carrier_assignment: List[int],
                 requests_disclosure_time: List[dt.datetime],
                 requests_x: List[float],
                 requests_y: List[float],
                 requests_revenue: List[float],
                 requests_service_duration: List[dt.timedelta],
                 requests_load: List[float],
                 request_time_window_open: List[dt.datetime],
                 request_time_window_close: List[dt.datetime],
                 carrier_depots_x: List[float],
                 carrier_depots_y: List[float],
                 carrier_depots_tw_open: List[dt.datetime],
                 carrier_depots_tw_close: List[dt.datetime],
                 duration_matrix,
                 distance_matrix):
        """

        :param max_tour_duration:
        :param distance_matrix:
        :param id_: unique identifier
        :param carriers_max_num_tours:
        :param max_vehicle_load:
        :param max_tour_length:
        :param requests: list of request indices
        :param requests_initial_carrier_assignment:
        :param requests_disclosure_time:
        :param requests_x:
        :param requests_y:
        :param requests_revenue:
        :param requests_service_duration:
        :param requests_load:
        :param request_time_window_open:
        :param request_time_window_close:
        :param carrier_depots_x:
        :param carrier_depots_y:
        :param carrier_depots_tw_open:
        :param carrier_depots_tw_close:
        :param duration_matrix:
        """
        # sanity checks:
        assert requests == sorted(requests)
        assert requests[0] == 0
        assert requests[-1] == len(requests) - 1
        assert len(carrier_depots_x) == len(carrier_depots_y)
        assert len(requests_x) == len(requests_y)
        assert all(ut.ACCEPTANCE_START_TIME <= t <= ut.EXECUTION_START_TIME for t in requests_disclosure_time)
        assert all(x <= max_vehicle_load for x in requests_load)
        assert all(ut.EXECUTION_START_TIME <= t <= ut.END_TIME for t in request_time_window_open)
        assert all(ut.EXECUTION_START_TIME <= t <= ut.END_TIME for t in request_time_window_close)

        self._id_ = id_
        self.meta = dict(
            (k.strip(), v if k == 't' else int(v.strip())) for k, v in (item.split('=') for item in id_.split('+')))
        self.num_carriers = len(carrier_depots_x)
        self.max_vehicle_load = max_vehicle_load
        self.max_tour_distance = max_tour_length
        self.max_tour_duration = max_tour_duration
        self.carriers_max_num_tours = carriers_max_num_tours
        self.requests = requests
        self.num_requests = len(self.requests)
        assert self.num_requests % self.num_carriers == 0
        self.num_requests_per_carrier = self.num_requests // self.num_carriers
        self.vertex_x_coords = [*carrier_depots_x, *requests_x]
        self.vertex_y_coords = [*carrier_depots_y, *requests_y]
        assert all(x in range(self.num_carriers) for x in requests_initial_carrier_assignment)
        self.request_to_carrier_assignment: List[int] = requests_initial_carrier_assignment
        self.request_disclosure_time: List[dt.datetime] = requests_disclosure_time
        self.vertex_revenue = [*[0] * self.num_carriers, *requests_revenue]
        self.vertex_load = [*[0] * self.num_carriers, *requests_load]
        self.vertex_service_duration = (*[dt.timedelta(0)] * self.num_carriers, *requests_service_duration)
        self.tw_open = [*carrier_depots_tw_open, *request_time_window_open]
        self.tw_close = [*carrier_depots_tw_close, *request_time_window_close]

        # need to ceil the durations due to floating point precision!
        self._travel_duration_matrix = np.array(duration_matrix)
        self._distance_matrix = np.array(distance_matrix)

        logger.debug(f'{id_}: created')

    pass

    def __repr__(self):
        return f'MDVRPTW Instance {self.id_}({len(self.requests)} customers, {self.num_carriers} carriers)'

    @property
    def id_(self):
        return self._id_

    def distance(self, i: Sequence[int], j: Sequence[int]) -> NoReturn:
        """
        returns the distance between pairs of elements in i and j. Think sum(distance(i[0], j[0]), distance(i[1], j[1]),
        ...)

        """
        d = 0
        for ii, jj in zip(i, j):
            d += self._distance_matrix[ii, jj]
        return d

    def travel_duration(self, i: Sequence[int], j: Sequence[int]):
        """
        returns the distance between pairs of elements in i and j. Think sum(distance(i[0], j[0]), distance(i[1], j[1]),
        ...)

        """
        d = dt.timedelta(0)
        for ii, jj in zip(i, j):
            d += self._travel_duration_matrix[ii, jj]
        return d

    def vertex_from_request(self, request: int) -> int:
        if request >= self.num_requests:
            raise IndexError
        return self.num_carriers + request

    def request_from_vertex(self, vertex: int):
        if vertex < self.num_carriers:
            raise IndexError(f'you provided vertex {vertex} but that is a depot vertex, not a request vertex')
        elif vertex >= self.num_carriers + self.num_requests:
            raise IndexError(
                f'you provided vertex {vertex} but there are only {self.num_carriers + 2 * self.num_requests} vertices')
        else:
            return vertex - self.num_carriers

    def coords(self, vertex: int):
        """returns a tuple of (x, y) coordinates for the vertex"""
        return ut.Coordinates(self.vertex_x_coords[vertex], self.vertex_y_coords[vertex])

    def assign_time_window(self, vertex: int, time_window: tw.TimeWindow):
        """
        changes the time window of a vertex
        """
        assert self.num_carriers <= vertex < self.num_carriers + self.num_requests
        self.tw_open[vertex] = time_window.open
        self.tw_close[vertex] = time_window.close

    def write(self, path: Path, delim=','):
        lines = [f'# VRP parameters: V = num of vehicles, L = max_load, T = max_tour_length']
        lines.extend([f'V{delim}{self.carriers_max_num_tours}',
                      f'L{delim}{self.max_vehicle_load}',
                      f'T{delim}{self.max_tour_distance}\n'])
        lines.extend(['# carrier depots: C x y',
                      '# one line per carrier, number of carriers defined by number of lines'])
        lines.extend([f'C{delim}{x}{delim}{y}'
                      for x, y in
                      zip(self.vertex_x_coords[:self.num_carriers], self.vertex_y_coords[:self.num_carriers])])
        lines.extend(['\n# requests: carrier_index delivery_x delivery_y revenue',
                      '# carrier_index = line index of carriers above'])
        for request in self.requests:
            lines.append(
                f'{self.request_to_carrier_assignment[request]}{delim}'
                f'{self.vertex_x_coords[request + self.num_carriers]}{delim}'
                f'{self.vertex_y_coords[request + self.num_carriers]}{delim}'
                f'{self.vertex_revenue[request + self.num_carriers]}'
            )

        lines.append(f'\n# travel duration in seconds. initial entries correspond to depots')

        for i in range(len(self._travel_duration_matrix)):
            lines.append(delim.join([str(x.total_seconds()) for x in self._travel_duration_matrix[i]]))

        with path.open('w') as f:
            f.writelines([l + '\n' for l in lines])

        pass


class MDPDPTWInstance:
    def __init__(self,
                 id_: str,
                 max_num_tours_per_carrier: int,
                 max_vehicle_load: float,
                 max_tour_length: float,
                 max_tour_duration: dt.timedelta,
                 requests: List[int],
                 requests_initial_carrier_assignment: List[int],
                 requests_disclosure_time: List[dt.datetime],
                 requests_pickup_x: List[float],
                 requests_pickup_y: List[float],
                 requests_delivery_x: List[float],
                 requests_delivery_y: List[float],
                 requests_revenue: List[float],
                 requests_pickup_service_duration: List[dt.timedelta],
                 requests_delivery_service_duration: List[dt.timedelta],
                 requests_pickup_load: List[float],
                 requests_delivery_load: List[float],
                 request_pickup_time_window_open: List[dt.datetime],
                 request_pickup_time_window_close: List[dt.datetime],
                 request_delivery_time_window_open: List[dt.datetime],
                 request_delivery_time_window_close: List[dt.datetime],
                 carrier_depots_x: List[float],
                 carrier_depots_y: List[float],
                 carrier_depots_tw_open: List[dt.datetime],
                 carrier_depots_tw_close: List[dt.datetime]):
        """
        Create an instance for the collaborative transportation network for attended home delivery

        :param max_tour_duration:
        :param id_: unique identifier
        """
        # sanity checks:
        assert len(carrier_depots_x) == len(carrier_depots_y)
        assert len(requests_pickup_x) == len(requests_pickup_y) == len(requests_delivery_x) == len(requests_delivery_y)
        assert all(ut.ACCEPTANCE_START_TIME <= t <= ut.EXECUTION_START_TIME for t in requests_disclosure_time)
        assert all(x <= max_vehicle_load for x in requests_pickup_load)
        assert all(ut.EXECUTION_START_TIME <= t <= ut.END_TIME for t in request_pickup_time_window_open)
        assert all(ut.EXECUTION_START_TIME <= t <= ut.END_TIME for t in request_pickup_time_window_close)

        self._id_ = id_
        self.meta = dict((k.strip(), int(v.strip())) for k, v in (item.split('=') for item in id_.split('+')))
        self.num_carriers = len(carrier_depots_x)
        self.vehicles_max_load = max_vehicle_load
        self.vehicles_max_travel_distance = max_tour_length
        self.max_tour_duration = max_tour_duration
        self.carriers_max_num_tours = max_num_tours_per_carrier
        self.requests = requests
        self.num_requests = len(self.requests)
        assert self.num_requests % self.num_carriers == 0
        self.num_requests_per_carrier = self.num_requests // self.num_carriers
        self.vertex_x_coords = [*carrier_depots_x, *requests_pickup_x, *requests_delivery_x]
        self.vertex_y_coords = [*carrier_depots_y, *requests_pickup_y, *requests_delivery_y]
        assert all(x in range(self.num_carriers) for x in requests_initial_carrier_assignment)
        self.request_to_carrier_assignment: List[int] = requests_initial_carrier_assignment
        self.request_disclosure_time: List[dt.datetime] = requests_disclosure_time
        self.vertex_revenue = [*[0] * (self.num_carriers + len(requests)), *requests_revenue]
        self.vertex_load = [*[0] * self.num_carriers, *requests_pickup_load, *requests_delivery_load]
        self.vertex_service_duration = (*[dt.timedelta(0)] * self.num_carriers,
                                        *requests_pickup_service_duration,
                                        *requests_delivery_service_duration)
        self.tw_open = [*carrier_depots_tw_open, *request_pickup_time_window_open, *request_delivery_time_window_open]
        self.tw_close = [*carrier_depots_tw_close, *request_pickup_time_window_close,
                         *request_delivery_time_window_close]

        # compute the distance and travel time matrix
        # need to ceil the distances due to floating point precision!
        self._distance_matrix = np.ceil(
            squareform(pdist(np.array(list(zip(self.vertex_x_coords, self.vertex_y_coords))), 'euclidean'))).astype(
            'int')
        self._travel_time_matrix = [[ut.travel_time(d) for d in x] for x in self._distance_matrix]

        logger.debug(f'{id_}: created')

    def __repr__(self):
        return f'Instance {self.id_} with {len(self.requests)} customers, {self.num_carriers} carriers'

    @property
    def id_(self):
        return self._id_

    def distance(self, i: Sequence[int], j: Sequence[int]):
        """
        returns the distance between pairs of elements in i and j. Think sum(distance(i[0], j[0]), distance(i[1], j[1]),
        ...)

        """
        d = 0
        for ii, jj in zip(i, j):
            d += self._distance_matrix[ii, jj]
        return d

    def travel_duration(self, i: Sequence[int], j: Sequence[int]):
        """
        returns the travel time between pairs of elements in i and j.
        Think sum(travel_time(i[0], j[0]), travel_time(i[1], j[1]), ...)

        :param i:
        :param j:
        :return:
        """
        t = dt.timedelta(0)
        for ii, jj in zip(i, j):
            t += self._travel_time_matrix[ii][jj]
        return t

    def pickup_delivery_pair(self, request: int) -> Tuple[int, int]:
        """returns a tuple of pickup & delivery vertex indices for the given request"""
        if request >= self.num_requests:
            raise IndexError(
                f'you asked for request {request} but instance {self.id_} only has {self.num_requests} requests')
        return self.num_carriers + request, self.num_carriers + self.num_requests + request

    def request_from_vertex(self, vertex: int):
        if vertex < self.num_carriers:
            raise IndexError(f'you provided vertex {vertex} but that is a depot vertex, not a request vertex')
        elif vertex >= self.num_carriers + 2 * self.num_requests:
            raise IndexError(
                f'you provided vertex {vertex} but there are only {self.num_carriers + 2 * self.num_requests} vertices')
        elif vertex <= self.num_carriers + self.num_requests - 1:  # pickup vertex
            return vertex - self.num_carriers
        else:  # delivery vertex
            return vertex - self.num_carriers - self.num_requests

    def coords(self, vertex: int):
        """returns a tuple of (x, y) coordinates for the vertex"""
        return ut.Coordinates(self.vertex_x_coords[vertex], self.vertex_y_coords[vertex])

    def vertex_type(self, vertex: int):
        if vertex < self.num_carriers:
            return "depot"
        elif vertex < self.num_carriers + self.num_requests:
            return "pickup"
        elif vertex < self.num_carriers + 2 * self.num_requests:
            return "delivery"
        else:
            raise IndexError(f'Vertex index {vertex} out of range')

    def assign_time_window(self, vertex: int, time_window: tw.TimeWindow):
        """
        changes the time window of a vertex
        """
        assert self.num_carriers <= vertex < self.num_carriers + self.num_requests * 2
        self.tw_open[vertex] = time_window.open
        self.tw_close[vertex] = time_window.close


def read_gansterer_hartl_mv(path: Path, num_carriers=3) -> MDPDPTWInstance:
    """read an instance file as used in (Gansterer,M., & Hartl,R.F. (2016). Request evaluation strategies
    for carriers in auction-based collaborations. https://doi.org/10.1007/s00291-015-0411-1).
    CAUTION:multiplies the max vehicle load by 10!
    """
    vrp_params = pd.read_csv(path, skiprows=1, nrows=3, delim_whitespace=True, header=None, squeeze=True, index_col=0)
    depots = pd.read_csv(path, skiprows=7, nrows=num_carriers, delim_whitespace=True, header=None, index_col=False,
                         usecols=[1, 2], names=['x', 'y'])
    cols = ['carrier_index', 'pickup_x', 'pickup_y', 'delivery_x', 'delivery_y', 'revenue', 'load']
    requests = pd.read_csv(path, skiprows=10 + num_carriers, delim_whitespace=True, names=cols, index_col=False,
                           float_precision='round_trip')
    requests['pickup_service_time'] = dt.timedelta(0)
    requests['delivery_service_time'] = dt.timedelta(0)

    requests['disclosure_time'] = None
    for carrier_id in range(requests['carrier_index'].max() + 1):
        assert carrier_id < 24
        requests.loc[requests.carrier_index == carrier_id, 'disclosure_time'] = list(
            ut.datetime_range(start=ut.ACCEPTANCE_START_TIME,
                              stop=ut.EXECUTION_START_TIME,
                              num=len(requests[requests.carrier_index == carrier_id]),
                              endpoint=False))

    return MDPDPTWInstance(id_=path.stem, max_num_tours_per_carrier=vrp_params['V'].tolist(),
                           max_vehicle_load=(vrp_params['L'] * ut.LOAD_CAPACITY_SCALING).tolist(),
                           max_tour_length=vrp_params['T'].tolist(), max_tour_duration=None,
                           requests=requests.index.tolist(),
                           requests_initial_carrier_assignment=requests['carrier_index'].tolist(),
                           requests_disclosure_time=requests['disclosure_time'].tolist(),
                           requests_pickup_x=(requests['pickup_x'] * ut.DISTANCE_SCALING).tolist(),
                           requests_pickup_y=(requests['pickup_y'] * ut.DISTANCE_SCALING).tolist(),
                           requests_delivery_x=(requests['delivery_x'] * ut.DISTANCE_SCALING).tolist(),
                           requests_delivery_y=(requests['delivery_y'] * ut.DISTANCE_SCALING).tolist(),
                           requests_revenue=(requests['revenue'] * ut.REVENUE_SCALING).tolist(),
                           requests_pickup_service_duration=[x.to_pytimedelta() for x in
                                                             requests['pickup_service_time']],
                           requests_delivery_service_duration=[x.to_pytimedelta() for x in
                                                               requests['delivery_service_time']],
                           requests_pickup_load=requests['load'].tolist(),
                           requests_delivery_load=(-requests['load']).tolist(),
                           request_pickup_time_window_open=[ut.EXECUTION_START_TIME for _ in range(len(requests) * 2)],
                           request_pickup_time_window_close=[ut.END_TIME for _ in range(len(requests) * 2)],
                           request_delivery_time_window_open=[ut.EXECUTION_START_TIME for _ in
                                                              range(len(requests) * 2)],
                           request_delivery_time_window_close=[ut.END_TIME for _ in range(len(requests) * 2)],
                           carrier_depots_x=(depots['x'] * ut.DISTANCE_SCALING).tolist(),
                           carrier_depots_y=(depots['y'] * ut.DISTANCE_SCALING).tolist(),
                           carrier_depots_tw_open=[ut.EXECUTION_START_TIME for _ in range(len(depots))],
                           carrier_depots_tw_close=[ut.END_TIME for _ in range(len(depots))])


def read_vienna_instance(path: Path) -> MDVRPTWInstance:
    if path.suffix == ".json":
        with open(path, 'r') as file:
            inst = dict(json.load(file))
        inst['request_disclosure_time'] = [dt.datetime.fromisoformat(x) for x in inst['request_disclosure_time']]
        inst['vertex_service_duration'] = [dt.timedelta(seconds=x) for x in inst['vertex_service_duration']]
        inst['tw_open'] = [dt.datetime.fromisoformat(x) for x in inst['tw_open']]
        inst['tw_close'] = [dt.datetime.fromisoformat(x) for x in inst['tw_close']]
        inst['_travel_duration_matrix'] = [[dt.timedelta(seconds=j) for j in i]
                                           for i in inst['_travel_duration_matrix']]
        inst['max_tour_duration'] = dt.timedelta(seconds=inst['max_tour_duration'])

        return MDVRPTWInstance(id_=inst['_id_'],
                               carriers_max_num_tours=inst['carriers_max_num_tours'],
                               max_vehicle_load=inst['max_vehicle_load'],
                               max_tour_length=inst['max_tour_distance'],
                               max_tour_duration=inst['max_tour_duration'],
                               requests=inst['requests'],
                               requests_initial_carrier_assignment=inst['request_to_carrier_assignment'],
                               requests_disclosure_time=inst['request_disclosure_time'],
                               requests_x=inst['vertex_x_coords'][inst['num_carriers']:],
                               requests_y=inst['vertex_y_coords'][inst['num_carriers']:],
                               requests_revenue=inst['vertex_revenue'][inst['num_carriers']:],
                               requests_service_duration=inst['vertex_service_duration'][inst['num_carriers']:],
                               requests_load=inst['vertex_load'][inst['num_carriers']:],
                               request_time_window_open=inst['tw_open'][inst['num_carriers']:],
                               request_time_window_close=inst['tw_close'][inst['num_carriers']:],
                               carrier_depots_x=inst['vertex_x_coords'][:inst['num_carriers']],
                               carrier_depots_y=inst['vertex_y_coords'][:inst['num_carriers']],
                               carrier_depots_tw_open=inst['tw_open'][:inst['num_carriers']],
                               carrier_depots_tw_close=inst['tw_close'][:inst['num_carriers']],
                               duration_matrix=inst['_travel_duration_matrix'],
                               distance_matrix=inst['_distance_matrix'])
    else:
        raise NotImplementedError(f'Reading files from {path.suffix} files is not supported yet')


if __name__ == '__main__':
    from pprint import pprint

    inst = read_vienna_instance(Path(
        'C:/Users/Elting/ucloud/PhD/02_Research/02_Collaborative Routing for Attended Home Deliveries/01_Code/data/Input/t=vienna+d=7+c=3+n=10+o=030+r=00.json'))
    print(inst)
    pprint(inst.__dict__)
    pass
