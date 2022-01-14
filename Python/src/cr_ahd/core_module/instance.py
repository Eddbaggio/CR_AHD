import datetime as dt
import json
import logging.config
import math
from pathlib import Path
from typing import Sequence, List, NoReturn

import numpy as np

import utility_module.utils as ut
from instance_module.vienna_data_handling import check_triangle_inequality
from tw_management_module import tw
from utility_module.datetime_rounding import ceil_timedelta
from utility_module.io import MyJSONEncoder

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

        # TODO need to ceil the durations due to floating point precision?!, Yes!!
        # self._travel_duration_matrix = np.array(duration_matrix)
        self._travel_duration_matrix = np.array([[ceil_timedelta(x, 's') for x in y] for y in duration_matrix])
        assert all(self._travel_duration_matrix.ravel() >= dt.timedelta(0))
        num_violations = check_triangle_inequality(self._travel_duration_matrix, True)
        assert num_violations == 0, f'{self.id_} violates triangle inequality in {num_violations} cases'

        self._travel_distance_matrix = np.array([[math.ceil(x) for x in y] for y in distance_matrix])
        assert all(self._travel_distance_matrix.ravel() >= 0)

        logger.debug(f'{id_}: created')

    pass

    def __repr__(self):
        return f'MDVRPTW Instance {self.id_}({len(self.requests)} customers, {self.num_carriers} carriers)'

    @property
    def id_(self):
        return self._id_

    def travel_distance(self, i: Sequence[int], j: Sequence[int]) -> NoReturn:
        """
        returns the distance between pairs of elements in i and j. Think sum(distance(i[0], j[0]), distance(i[1], j[1]),
        ...)

        """
        d = 0
        for ii, jj in zip(i, j):
            d += self._travel_distance_matrix[ii, jj]
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

    def time_window(self, vertex: int):
        return ut.TimeWindow(self.tw_open[vertex], self.tw_close[vertex])

    def vertex_type(self, vertex: int):
        if vertex < self.num_carriers:
            return 'depot'
        elif vertex < self.num_carriers + self.num_requests:
            return 'pickup'
        else:
            raise IndexError(f'Vertex index {vertex} out of range')

    def assign_time_window(self, vertex: int, time_window: tw.TimeWindow):
        """
        changes the time window of a vertex
        """
        assert self.num_carriers <= vertex < self.num_carriers + self.num_requests
        self.tw_open[vertex] = time_window.open
        self.tw_close[vertex] = time_window.close

    def write_json(self, path:Path):
        with open(path, 'w') as file:
            json.dump(self.__dict__, file, cls=MyJSONEncoder, indent=4)

    def write_delim(self, path: Path, delim=','):
        """
        it is much easier to read instances from json files

        :param path:
        :param delim:
        :return:
        """
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

        lines.append(f'\n# travel distance in meters. initial entries correspond to depots')

        for i in range(len(self._travel_distance_matrix)):
            lines.append(delim.join([str(x) for x in self._travel_distance_matrix[i]]))

        with path.open('w') as f:
            f.writelines([l + '\n' for l in lines])

        pass


def read_vienna_instance(path: Path) -> MDVRPTWInstance:
    if path.suffix == ".json":
        with open(path, 'r') as file:
            inst = dict(json.load(file))
        inst['request_disclosure_time'] = [dt.datetime.fromisoformat(x) for x in inst['request_disclosure_time']]
        inst['vertex_service_duration'] = [dt.timedelta(seconds=x) for x in inst['vertex_service_duration']]
        inst['tw_open'] = [dt.datetime.fromisoformat(x) for x in inst['tw_open']]
        inst['tw_close'] = [dt.datetime.fromisoformat(x) for x in inst['tw_close']]
        inst['_travel_duration_matrix'] = [[dt.timedelta(seconds=y) for y in x]
                                           for x in inst['_travel_duration_matrix']]
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
                               distance_matrix=inst['_travel_distance_matrix'])
    else:
        raise NotImplementedError(f'Reading files from {path.suffix} files is not supported yet')


if __name__ == '__main__':
    from pprint import pprint

    inst = read_vienna_instance(Path(
        'C:/Users/Elting/ucloud/PhD/02_Research/02_Collaborative Routing for Attended Home Deliveries/01_Code/data/Input/t=vienna+d=7+c=3+n=10+o=030+r=00.json'))
    print(inst)
    pprint(inst.__dict__)
    pass
