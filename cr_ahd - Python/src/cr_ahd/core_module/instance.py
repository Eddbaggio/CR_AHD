import datetime as dt
import json
import logging.config
from pathlib import Path
from typing import Tuple, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

import src.cr_ahd.utility_module.utils as ut

logger = logging.getLogger(__name__)


class PDPInstance:
    def __init__(self,
                 id_: str,
                 max_num_tours_per_carrier: int,
                 max_vehicle_load: float,
                 max_tour_length: float,
                 requests: Sequence,
                 requests_initial_carrier_assignment: Sequence,
                 requests_pickup_x: Sequence,
                 requests_pickup_y: Sequence,
                 requests_delivery_x: Sequence,
                 requests_delivery_y: Sequence,
                 requests_revenue: Sequence,
                 requests_pickup_service_time: Sequence,
                 requests_delivery_service_time: Sequence,
                 requests_pickup_load: Sequence,
                 requests_delivery_load: Sequence,
                 carrier_depots_x: Sequence,
                 carrier_depots_y: Sequence,
                 ):
        """
        Create an instance for the collaborative transportation network for attended home delivery

        :param id_: unique identifier
        """
        self._id_ = id_
        self.meta = dict((k.strip(), int(v.strip()))
                         for k, v in (item.split('=')
                                      for item in id_.split('+')))
        self.num_depots = len(carrier_depots_x)
        self.num_carriers = len(carrier_depots_x)
        self.vehicles_max_load = max_vehicle_load
        self.vehicles_max_travel_distance = max_tour_length
        self.carriers_max_num_tours = max_num_tours_per_carrier
        self.requests = requests
        self.x_coords = [*carrier_depots_x, *requests_pickup_x, *requests_delivery_x]
        self.y_coords = [*carrier_depots_y, *requests_pickup_y, *requests_delivery_y]
        self.request_to_carrier_assignment = requests_initial_carrier_assignment
        self.revenue = [*[0] * (self.num_depots + len(requests)), *requests_revenue]
        self.load = [*[0] * self.num_depots, *requests_pickup_load, *requests_delivery_load]
        self.service_duration = [*[dt.timedelta(0)] * self.num_depots, *requests_pickup_service_time,
                                 *requests_delivery_service_time]

        # compute the distance matrix
        # need to round the distances due to floating point precision!
        self._distance_matrix = np.ceil(
            squareform(pdist(np.array(list(zip(self.x_coords, self.y_coords))), 'euclidean')))
        # self._distance_matrix = squareform(pdist(np.array(list(zip(self.x_coords, self.y_coords))), 'euclidean'))
        logger.debug(f'{id_}: created')

    def __str__(self):
        return f'Instance {self.id_} with {len(self.requests)} customers, {self.num_carriers} carriers and {self.num_depots} depots'

    @property
    def id_(self):
        return self._id_

    @property
    def num_requests(self):
        """The number of requests"""
        return len(self.requests)

    def distance(self, i: Sequence[int], j: Sequence[int]):
        """
        returns the distance between pairs of elements in i and j. Think sum(distance(i[0], j[0]), distance(i[1], j[1]),
        ...)

        """
        d = 0
        for ii, jj in zip(i, j):
            d += self._distance_matrix[ii, jj]
        return d

    def pickup_delivery_pair(self, request: int) -> Tuple[int, int]:
        """returns a tuple of pickup & delivery vertex indices for the given request"""
        if request >= self.num_requests:
            raise IndexError(
                f'you asked for request {request} but instance {self.id_} only has {self.num_requests} requests')
        return self.num_depots + request, self.num_depots + self.num_requests + request

    def request_from_vertex(self, vertex: int):
        if vertex < self.num_depots:
            raise IndexError(f'you provided vertex {vertex} but that is a depot vertex, not a request vertex')
        elif vertex >= self.num_depots + 2 * self.num_requests:
            raise IndexError(
                f'you provided vertex {vertex} but there are only {self.num_depots + 2 * self.num_requests} vertices')
        elif vertex <= self.num_depots + self.num_requests - 1:  # pickup vertex
            return vertex - self.num_depots
        else:  # delivery vertex
            return vertex - self.num_depots - self.num_requests

    def coords(self, vertex: int):
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
        file_name = ut.unique_path(file_name.parent, file_name.stem + '_#{:03d}' + '.json')
        file_name.parent.mkdir(parents=True, exist_ok=True)
        with open(file_name, mode='w') as write_file:
            json.dump(self.to_dict(), write_file, indent=4, cls=ut.MyJSONEncoder)
        # LOGGER.debug(f'{self.id_}: wrote instance to json at {file_name}')
        return file_name

    def vertex_type(self, vertex: int):
        if vertex < self.num_depots:
            return "depot"
        elif vertex < self.num_depots + self.num_requests:
            return "pickup"
        elif vertex < self.num_depots + 2 * self.num_requests:
            return "delivery"
        else:
            raise IndexError(f'Vertex index {vertex} out of range')


def read_gansterer_hartl_mv(path: Path, num_carriers=3) -> PDPInstance:
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
    requests['service_time'] = dt.timedelta(0)
    return PDPInstance(path.stem,
                       vrp_params['V'].tolist(),
                       (vrp_params['L'] * 10).tolist(),  # todo can i solve the problems without *10 vehicle capacity?
                       vrp_params['T'].tolist(),
                       requests.index.tolist(),
                       requests['carrier_index'].tolist(),
                       requests['pickup_x'].tolist(),
                       requests['pickup_y'].tolist(),
                       requests['delivery_x'].tolist(),
                       requests['delivery_y'].tolist(),
                       requests['revenue'].tolist(),
                       requests['service_time'].tolist(),
                       requests['service_time'].tolist(),
                       requests['load'].tolist(),
                       (-requests['load']).tolist(),
                       depots['x'].tolist(),
                       depots['y'].tolist(),
                       )


def read_gansterer_hartl_sv(path: Path) -> PDPInstance:
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
    return PDPInstance(path.stem, requests, depots, 1, float('inf'), float('inf'))


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
