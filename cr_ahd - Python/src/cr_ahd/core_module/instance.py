import datetime as dt
import json
import logging.config
from pathlib import Path
from typing import Tuple, Sequence, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

import tw_management_module.tw
import utility_module.io
import utility_module.utils as ut

logger = logging.getLogger(__name__)


class MDPDPTWInstance:
    def __init__(self,
                 id_: str,
                 max_num_tours_per_carrier: int,
                 max_vehicle_load: float,
                 max_tour_length: float,
                 requests: List[int],
                 requests_initial_carrier_assignment: List[int],
                 requests_pickup_x: List[float],
                 requests_pickup_y: List[float],
                 requests_delivery_x: List[float],
                 requests_delivery_y: List[float],
                 requests_revenue: List[float],
                 requests_pickup_service_time: List[dt.timedelta],
                 requests_delivery_service_time: List[dt.timedelta],
                 requests_pickup_load: List[float],
                 requests_delivery_load: List[float],
                 request_pickup_time_window_open: List[dt.datetime],
                 request_pickup_time_window_close: List[dt.datetime],
                 request_delivery_time_window_open: List[dt.datetime],
                 request_delivery_time_window_close: List[dt.datetime],
                 carrier_depots_x: List[float],
                 carrier_depots_y: List[float],
                 carrier_depots_tw_open: List[dt.datetime],
                 carrier_depots_tw_close: List[dt.datetime],
                 ):
        """
        Create an instance for the collaborative transportation network for attended home delivery

        :param id_: unique identifier
        """
        self._id_ = id_
        self.meta = dict((k.strip(), int(v.strip())) for k, v in (item.split('=') for item in id_.split('+')))
        self.num_carriers = len(carrier_depots_x)
        self.num_carriers = len(carrier_depots_x)
        self.vehicles_max_load = max_vehicle_load
        self.vehicles_max_travel_distance = max_tour_length
        self.carriers_max_num_tours = max_num_tours_per_carrier
        self.requests = requests
        self.num_requests = len(self.requests)
        assert self.num_requests % self.num_carriers == 0
        self.num_requests_per_carrier = self.num_requests // self.num_carriers
        self.x_coords = [*carrier_depots_x, *requests_pickup_x, *requests_delivery_x]
        self.y_coords = [*carrier_depots_y, *requests_pickup_y, *requests_delivery_y]
        self.request_to_carrier_assignment: List[int] = requests_initial_carrier_assignment
        self.vertex_revenue = [*[0] * (self.num_carriers + len(requests)), *requests_revenue]
        self.vertex_load = [*[0] * self.num_carriers, *requests_pickup_load, *requests_delivery_load]
        self.vertex_service_duration = (*[dt.timedelta(0)] * self.num_carriers,
                                        *requests_pickup_service_time,
                                        *requests_delivery_service_time)
        self.tw_open = [*carrier_depots_tw_open, *request_pickup_time_window_open, *request_delivery_time_window_open]
        self.tw_close = [*carrier_depots_tw_close, *request_pickup_time_window_close,
                         *request_delivery_time_window_close]

        # compute the distance and travel time matrix
        # need to ceil the distances due to floating point precision!
        self._distance_matrix = np.ceil(
            squareform(pdist(np.array(list(zip(self.x_coords, self.y_coords))), 'euclidean'))).astype('int')
        self._travel_time_matrix = [[ut.travel_time(d) for d in x] for x in self._distance_matrix]

        logger.debug(f'{id_}: created')

    def __str__(self):
        return f'Instance {self.id_} with {len(self.requests)} customers, {self.num_carriers} carriers'

    # TODO implement deepcopy!
    # def __deepcopy__(self, memodict={}):
    #     pass

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
        return ut.Coordinates(self.x_coords[vertex], self.y_coords[vertex])

    def write_to_json(self):
        """
        Write the instance's data in a json format

        :return: the file path as a pathlib.Path object
        """
        file_name = ut.path_input_custom.joinpath(self.solomon_base, self.id_)
        # check if any customers are assigned to carriers already and set the file name
        if self.assigned_requests():
            file_name = file_name.with_name(file_name.stem + '_ass')
        file_name = utility_module.io.unique_path(file_name.parent, file_name.stem + '_#{:03d}' + '.json')
        file_name.parent.mkdir(parents=True, exist_ok=True)
        with open(file_name, mode='w') as write_file:
            json.dump(self.to_dict(), write_file, indent=4, cls=utility_module.io.MyJSONEncoder)
        # LOGGER.debug(f'{self.id_}: wrote instance to json at {file_name}')
        return file_name

    def vertex_type(self, vertex: int):
        if vertex < self.num_carriers:
            return "depot"
        elif vertex < self.num_carriers + self.num_requests:
            return "pickup"
        elif vertex < self.num_carriers + 2 * self.num_requests:
            return "delivery"
        else:
            raise IndexError(f'Vertex index {vertex} out of range')

    def assign_time_window(self, vertex: int, time_window: tw_management_module.tw.TimeWindow):
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
    return MDPDPTWInstance(id_=path.stem,
                           max_num_tours_per_carrier=vrp_params['V'].tolist(),
                           max_vehicle_load=(vrp_params['L'] * ut.LOAD_CAPACITY_SCALING).tolist(),
                           max_tour_length=vrp_params['T'].tolist(),
                           requests=requests.index.tolist(),
                           requests_initial_carrier_assignment=requests['carrier_index'].tolist(),
                           requests_pickup_x=(requests['pickup_x'] * ut.DISTANCE_SCALING).tolist(),
                           requests_pickup_y=(requests['pickup_y'] * ut.DISTANCE_SCALING).tolist(),
                           requests_delivery_x=(requests['delivery_x'] * ut.DISTANCE_SCALING).tolist(),
                           requests_delivery_y=(requests['delivery_y'] * ut.DISTANCE_SCALING).tolist(),
                           requests_revenue=(requests['revenue'] * ut.REVENUE_SCALING).tolist(),
                           requests_pickup_service_time=[x.to_pytimedelta() for x in requests['pickup_service_time']],
                           requests_delivery_service_time=[x.to_pytimedelta() for x in
                                                           requests['delivery_service_time']],
                           requests_pickup_load=requests['load'].tolist(),
                           requests_delivery_load=(-requests['load']).tolist(),
                           request_pickup_time_window_open=[ut.START_TIME for _ in range(len(requests) * 2)],
                           request_pickup_time_window_close=[ut.END_TIME for _ in range(len(requests) * 2)],
                           request_delivery_time_window_open=[ut.START_TIME for _ in range(len(requests) * 2)],
                           request_delivery_time_window_close=[ut.END_TIME for _ in range(len(requests) * 2)],
                           carrier_depots_x=(depots['x'] * ut.DISTANCE_SCALING).tolist(),
                           carrier_depots_y=(depots['y'] * ut.DISTANCE_SCALING).tolist(),
                           carrier_depots_tw_open=[ut.START_TIME for _ in range(len(depots))],
                           carrier_depots_tw_close=[ut.END_TIME for _ in range(len(depots))]
                           )


def read_gansterer_hartl_sv(path: Path) -> MDPDPTWInstance:
    """read an instance file as used in (Gansterer,M., & Hartl,R.F. (2016). Request evaluation strategies
    for carriers in auction-based collaborations. https://doi.org/10.1007/s00291-015-0411-1).
    """
    raise NotImplementedError
    depots = pd.read_csv(path, skiprows=2, nrows=3, delim_whitespace=True, header=None, index_col=False,
                         usecols=[1, 2], names=['x', 'y'])
    cols = ['carrier_index', 'pickup_x', 'pickup_y', 'delivery_x', 'delivery_y', 'revenue', 'load']
    requests = pd.read_csv(path, skiprows=8, delim_whitespace=True, names=cols, index_col=False,
                           float_precision='round_trip')
    requests['service_time'] = dt.timedelta(0)
    return MDPDPTWInstance(path.stem, requests, depots, 1, float('inf'), float('inf'))


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
