import datetime as dt
import json
import logging.config
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

import src.cr_ahd.utility_module.utils as ut

logger = logging.getLogger(__name__)


class PDPInstance:
    def __init__(self,
                 id_: str,
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
        self.meta = dict((k.strip(), int(v.strip()))
                         for k, v in (item.split('=')
                         for item in id_.split('+')))
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
            [*pd.Series([pd.Timedelta(0)] * self.num_carriers), *requests['service_time'], *requests['service_time']])

        # compute the distance matrix
        request_coordinates = pd.concat([
            requests[['pickup_x', 'pickup_y']].rename(columns={'pickup_x': 'x', 'pickup_y': 'y'}),
            requests[['delivery_x', 'delivery_y']].rename(columns={'delivery_x': 'x', 'delivery_y': 'y'})],
            keys=['pickup', 'delivery']
        ).swaplevel()
        self._distance_matrix = squareform(pdist(np.concatenate([carrier_depots, request_coordinates]), 'euclidean'))
        # self.df = requests  # store the data as pd.DataFrame as well?
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

    def distance(self, i: List[int], j: List[int]):
        d = 0
        for ii in i:
            for jj in j:
                d += self._distance_matrix[ii, jj]
        return d

    def pickup_delivery_pair(self, request: int) -> Tuple[int, int]:
        """returns a tuple of pickup & delivery vertex indices for the given request"""
        return self.num_carriers + request, self.num_carriers + self.num_requests + request

    def request_from_vertex(self, vertex: int):
        assert vertex >= self.num_carriers, f'vertex {vertex} does not belong to a request but is a depot vertex'
        if vertex <= self.num_carriers + self.num_requests - 1:  # pickup vertex
            return vertex - self.num_carriers
        else:  # delivery vertex
            return vertex - self.num_carriers - self.num_requests

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
    '''

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


def read_gansterer_hartl_mv(path: Path, num_carriers=3) -> PDPInstance:
    """read an instance file as used in (Gansterer,M., & Hartl,R.F. (2016). Request evaluation strategies
    for carriers in auction-based collaborations. https://doi.org/10.1007/s00291-015-0411-1).
    multiplies the max vehicle load by 10!
    """
    vrp_params = pd.read_csv(path, skiprows=1, nrows=3, delim_whitespace=True, header=None, squeeze=True, index_col=0)
    depots = pd.read_csv(path, skiprows=7, nrows=num_carriers, delim_whitespace=True, header=None, index_col=False,
                         usecols=[1, 2], names=['x', 'y'])
    cols = ['carrier_index', 'pickup_x', 'pickup_y', 'delivery_x', 'delivery_y', 'revenue', 'load']
    requests = pd.read_csv(path, skiprows=10 + num_carriers, delim_whitespace=True, names=cols, index_col=False,
                           float_precision='round_trip')
    requests['service_time'] = dt.timedelta(0)
    return PDPInstance(path.stem, requests, depots, vrp_params['V'], vrp_params['L'] * 10, vrp_params['T'])


def read_gansterer_hartl_sv(path: Path) -> PDPInstance:
    """read an instance file as used in (Gansterer,M., & Hartl,R.F. (2016). Request evaluation strategies
    for carriers in auction-based collaborations. https://doi.org/10.1007/s00291-015-0411-1).
    """
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
