import datetime as dt
import logging.config
from typing import List
import numpy as np

import src.cr_ahd.utility_module.utils as ut

logger = logging.getLogger(__name__)


class Tour:
    def __init__(self, id_, instance, solution):
        self.id_ = id_
        self._routing_sequence = []  # sequence of vertices
        self._travel_distance_sequence: List[float] = []  # sequence of arc distance costs
        self._travel_duration_sequence: List[dt.timedelta] = []  # sequence of arc duration costs
        self._load_sequence: List[float] = []  # the delta of the fill level of the vehicle (positive or negative)
        self._revenue_sequence: List[float] = []  # the collected revenue
        self._arrival_schedule: List[dt.datetime] = []  # arrival time at each stop
        self._service_schedule: List[dt.datetime] = []  # start of service time at each stop

        # initialize depot to depot tour
        self.insert_and_update(instance, solution, [0], [0])
        self.insert_and_update(instance, solution, [1], [0])

    def __str__(self):
        arrival_schedule = []
        service_schedule = []
        for i in range(len(self)):
            if self.arrival_schedule[i] is None:
                arrival_schedule.append(None)
                service_schedule.append(None)
            else:
                arrival_schedule.append(self.arrival_schedule[i])
                service_schedule.append(self.service_schedule[i])
        distance = round(self.sum_travel_distance, 2)
        return f'ID:\t\t\t{self.id_}\nSequence:\t{self.routing_sequence}\nArrival:\t{arrival_schedule}\n' \
               f'Service:\t{service_schedule}\nDistance:\t\t{distance}\nDuration:\t\t{self.sum_travel_duration} '

    def __len__(self):
        return len(self.routing_sequence)

    @property
    def routing_sequence(self):
        """immutable sequence of routed vertices. can only be modified by inserting"""
        return tuple(self._routing_sequence)

    @property
    def num_routing_stops(self):
        return len(self)

    @property
    def travel_distance_sequence(self):
        return tuple(self._travel_distance_sequence)

    @property
    def sum_travel_distance(self) -> float:
        return sum(self._travel_distance_sequence)

    @property
    def cumsum_travel_distance(self):
        return np.cumsum(self.travel_distance_sequence)

    @property
    def travel_duration_sequence(self):
        return tuple(self._travel_duration_sequence)

    @property
    def sum_travel_duration(self) -> dt.timedelta:
        return sum(self._travel_duration_sequence, dt.timedelta())

    @property
    def cumsum_travel_duration(self):
        return np.cumsum(self.travel_duration_sequence)

    @property
    def load_sequence(self):
        return tuple(self._load_sequence)

    @property
    def sum_load(self):
        return sum(self.load_sequence)

    @property
    def cumsum_load(self):
        return np.cumsum(self.load_sequence)

    @property
    def revenue_sequence(self):
        return self._revenue_sequence

    @property
    def sum_revenue(self):
        return sum(self.revenue_sequence)

    @property
    def cumsum_revenue(self):
        return np.cumsum(self.revenue_sequence)

    @property
    def arrival_schedule(self):
        return tuple(self._arrival_schedule)

    @property
    def service_schedule(self):
        return tuple(self._service_schedule)

    def as_dict(self):
        return {
            'routing_sequence': self.routing_sequence,
            'travel_distance_sequence': self.travel_distance_sequence,
            'travel_duration_sequence': self.travel_duration_sequence,
            'arrival_schedule': self.arrival_schedule,
            'service_schedule': self.service_schedule,
            'load_schedule': self.load_sequence,
            'revenue_schedule': self.revenue_sequence,
        }

    def summary(self):
        return {
            # 'id_': self.id_,
            'num_routing_stops': self.num_routing_stops,
            'sum_travel_distance': self.sum_travel_distance,
            'sum_travel_duration': self.sum_travel_duration,
            'sum_load': self.sum_load,  # should always be 0
            'sum_revenue': self.sum_revenue,
        }

    def _insert_no_update(self, indices: List[int], vertices: List[int]):
        """highly discouraged to use this as it does not ensure feasibility of the route! use insert_and_update instead.
        use this only if infeasible route is acceptable, as e.g. reversing a section where intermediate states of the
        reversal may be infeasible"""
        for index, vertex in zip(indices, vertices):
            self._routing_sequence.insert(index, vertex)
            self._travel_duration_sequence.insert(index, dt.timedelta(0))
            self._travel_distance_sequence.insert(index, 0)
            if len(self) <= 1:
                self._arrival_schedule.insert(index, ut.START_TIME)
                self._service_schedule.insert(index, ut.START_TIME)
                self._load_sequence.insert(index, 0)
            else:
                self._arrival_schedule.insert(index, None)
                self._service_schedule.insert(index, None)
                self._load_sequence.insert(index, None)
            logger.debug(f'{vertex} inserted into {self.id_} at index {index}')
        pass

    def insert_and_update(self, instance, solution,
                          indices: List[int], vertices: List[int]):
        """
         inserts a vertex BEFORE the specified index and resets/deletes all cost and schedules. If insertion is
         infeasible due to time window constraints, it will undo the insertion and raise InsertionError

        :param solution:
        :param instance:
        :param indices: index before which the given vertex/request is to be inserted
        :param vertices: vertex to insert into the tour
        """
        assert all(indices[i] <= indices[i + 1] for i in range(len(indices) - 1))  # assure that indices are sorted

        try:
            self._insert_no_update(indices, vertices)
            if len(self) > 1:
                self.update_cost_and_schedules_from(instance, solution, indices[0])
        except ut.InsertionError as e:
            self.pop_and_update(instance, solution, indices)
            raise e
        pass

    def _pop_no_update(self, index: int):
        """highly discouraged to use this as it does not ensure feasibility. only use for intermediate states that
        allow infeasible states such as for reversing a section! use pop_and_update instead"""
        popped = self._routing_sequence.pop(index)
        self._travel_duration_sequence.pop(index)
        self._travel_distance_sequence.pop(index)
        self._arrival_schedule.pop(index)
        self._service_schedule.pop(index)
        self._load_sequence.pop(index)
        logger.debug(f'{popped} popped from {self.id_} at index {index}')
        return popped

    def pop_and_update(self, instance, solution, indices: List[int]):
        """removes the vertex at the index position from the tour and resets all cost and schedules"""
        popped = []
        for index in indices:
            popped.append(self._pop_no_update(index))
        self.update_cost_and_schedules_from(instance, solution, indices[0])
        return popped

    def update_cost_and_schedules_from(self, instance,
                                       solution,
                                       index: int = 1):
        """update schedules from given index to end of routing sequence. index should be the same as the insertion
        or removal index. """
        for pos in range(index, len(self)):
            vertex: int = self.routing_sequence[pos]
            predecessor_vertex: int = self.routing_sequence[pos - 1]
            travel_dist = instance.distance(predecessor_vertex, vertex)
            cumul_travel_dist = sum(self.travel_distance_sequence[:pos]) + travel_dist
            arrival = self.service_schedule[pos - 1] + ut.travel_time(travel_dist)
            cumul_load = sum(self.load_sequence[:pos]) + instance.load[vertex]
            try:
                # check the feasibility constraints
                assert cumul_travel_dist < instance.vehicles_max_travel_distance, f'{cumul_travel_dist} too long'
                assert arrival <= solution.tw_close[vertex], f'{arrival} too late'
                assert cumul_load <= instance.vehicles_max_load, f'{cumul_load} to high'
                self._travel_distance_sequence[pos] = travel_dist
                self._travel_duration_sequence[pos] = ut.travel_time(travel_dist)
                self._arrival_schedule[pos] = arrival
                self._service_schedule[pos] = max(arrival, solution.tw_open[vertex])
                self._load_sequence[pos] = instance.load[vertex]
            except AssertionError:
                logger.debug(f'{self.id_} update from index {index} failed at position {pos} with vertex {vertex}: '
                             f'cumul_travel_dist: {cumul_travel_dist}, max cumul_travel_dist: {instance.vehicles_max_travel_distance} '
                             f'arrival: {arrival}, tw_close: {solution.tw_close[vertex]}; '
                             f'cumul_load: {cumul_load}, max vehicle cumul_load: {instance.vehicles_max_load} '
                             )
                raise ut.InsertionError('One of the feasibility constraints is not satisfied. Check feasibility before '
                                        'insertion using Tour.insertion_feasibility_check()', '')
        logger.debug(f'{self.id_} updated from index {index}')
        pass

    def reverse_section(self, instance, solution, i, j):
        """
        reverses a section of the route from index i to index j-1. If reversal is infeasible, will raise InsertionError
        (and undoes the attempted reversal)

        Example: \n
        >> tour.sequence = [0, 1, 2, 3, 4, 0] \n
        >> tour.reverse_section(1, 4) \n
        >> print (tour.sequence) \n
        >> [0, 3, 2, 1, 4, 0]
        """
        for k in range(1, j - i):
            popped = self._pop_no_update(i)
            self._insert_no_update(j - k, popped)
        try:
            self.update_cost_and_schedules_from(instance, solution, i)  # maybe the new routing sequence is infeasible
        except ut.InsertionError as e:
            self.reverse_section(instance, solution, i, j)  # undo all the reversal
            raise e

    def insertion_distance_cost(self, instance, insertion_positions: List[int],
                                insertion_vertices: List[int]):
        """
        Does NOT perform a feasibility check!

        :param instance:
        :param insertion_positions:
        :param insertion_vertices:
        :return:
        """
        tmp_routing_sequence = list(self.routing_sequence)
        delta = 0
        for pos, vertex in zip(insertion_positions, insertion_vertices):
            predecessor_vertex = tmp_routing_sequence[pos - 1]
            successor_vertex = tmp_routing_sequence[pos]
            tmp_routing_sequence.insert(pos, vertex)
            delta += instance.distance(predecessor_vertex, vertex) + \
                     instance.distance(vertex, successor_vertex) - \
                     instance.distance(predecessor_vertex, successor_vertex)

        return delta

    def insertion_feasibility_check(self, instance,
                                    solution,
                                    insertion_positions: List[int],
                                    insertion_vertices: List[int]):
        """
        check Time Window, Maximum Tour Length and Maximum Vehicle Load constraints for the route IF the
        insertion_vertices were inserted at the insertion_positions. Insertion is not actually performed!

        :param instance:
        :param solution:
        :param insertion_positions:
        :param insertion_vertices:
        :return: True if all constraints are satisfied given the insertion; False otherwise
        """
        assert all(insertion_positions[i] <= insertion_positions[i + 1] for i in range(len(insertion_positions) - 1))
        # create a temporary routing sequence to loop over that contains the new vertices
        tmp_routing_sequence = list(self.routing_sequence)
        for pos, vertex in zip(insertion_positions, insertion_vertices):
            tmp_routing_sequence.insert(pos, vertex)
        tour_length = sum(self.travel_distance_sequence[:insertion_positions[0]])
        arrival_time = self.arrival_schedule[insertion_positions[0] - 1]
        load = self.load_sequence[insertion_positions[0] - 1]

        # iterate over the temporary tour and check all constraints
        for pos in range(insertion_positions[0], len(tmp_routing_sequence)):
            vertex: int = tmp_routing_sequence[pos]
            predecessor_vertex: int = tmp_routing_sequence[pos - 1]
            travel_distance = instance.distance(predecessor_vertex, vertex)
            # check max tour distance
            tour_length += travel_distance
            if tour_length > instance.vehicles_max_travel_distance:
                return False
            # check time windows
            arrival_time = arrival_time + instance.service_time[predecessor_vertex] + ut.travel_time(travel_distance)
            if arrival_time > solution.tw_close[vertex]:
                return False
            # check max vehicle load
            load += instance.load[vertex]
            if load > instance.vehicles_max_load:
                return False

        return True
