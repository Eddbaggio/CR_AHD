import datetime as dt
import logging.config
from typing import List, Union, Tuple, Iterable, Sequence
import numpy as np

import src.cr_ahd.utility_module.utils as ut

logger = logging.getLogger(__name__)


class Tour:
    def __init__(self, id_, instance, solution, depot_index):
        """

        :param id_:
        :param instance:
        :param solution:
        :param depot_index: may be different to the id! id's can exist twice temporarily if a carrier is copied!
        """
        logger.debug(f'Initializing tour {id_}')

        self.id_ = id_
        self._routing_sequence = []  # sequence of vertices
        self._travel_distance_sequence: List[float] = []  # sequence of arc distance costs
        self._travel_duration_sequence: List[dt.timedelta] = []  # sequence of arc duration costs
        self._load_sequence: List[float] = []  # the delta of the fill level of the vehicle (positive or negative)
        self._revenue_sequence: List[float] = []  # the collected revenue
        self._arrival_schedule: List[dt.datetime] = []  # arrival time at each stop
        self._service_schedule: List[dt.datetime] = []  # start of service time at each stop

        # initialize depot to depot tour
        self.insert_and_update(instance, solution, [1], [depot_index])
        self.insert_and_update(instance, solution, [0], [depot_index])

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
    def profit_sequence(self):
        raise NotImplementedError

    @property
    def sum_profit(self):
        return self.sum_revenue - self.sum_travel_distance

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
            'sum_profit': self.sum_profit,
            'num_routing_stops': self.num_routing_stops,
            'sum_travel_distance': self.sum_travel_distance,
            'sum_travel_duration': self.sum_travel_duration,
            'sum_load': self.sum_load,  # should always be 0
            'sum_revenue': self.sum_revenue,
        }

    def _insert_no_update(self, index: int, vertex: int):
        """highly discouraged to use this as it does not ensure feasibility of the route! use insert_and_update instead.
        use this only if infeasible route is acceptable, as e.g. reversing a section where intermediate states of the
        reversal may be infeasible"""
        self._routing_sequence.insert(index, vertex)
        self._travel_duration_sequence.insert(index, dt.timedelta(0))
        self._travel_distance_sequence.insert(index, 0)
        if len(self) <= 1:
            self._arrival_schedule.insert(index, ut.START_TIME)
            self._service_schedule.insert(index, ut.START_TIME)
            self._load_sequence.insert(index, 0)
            self._revenue_sequence.insert(index, 0)
        else:
            self._arrival_schedule.insert(index, None)
            self._service_schedule.insert(index, None)
            self._load_sequence.insert(index, None)
            self._revenue_sequence.insert(index, None)

        logger.debug(f'{vertex} inserted into {self.id_} at index {index}')
        pass

    def insert_and_update(self, instance, solution, indices: Sequence[int], vertices: Sequence[int]):
        """
         inserts a vertex BEFORE the specified index and updates all sequences and schedules.

        :param solution:
        :param instance:
        :param indices: index before which the given vertex is to be inserted
        :param vertices: vertex to insert into the tour
        """
        assert all(indices[i] <= indices[i + 1] for i in range(len(indices) - 1))  # assure that indices are sorted

        for index, vertex in zip(indices, vertices):
            self._insert_no_update(index, vertex)
        if len(self) > 1:
            self.update_sequences_and_schedules(instance, solution, indices[0])

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
        self._revenue_sequence.pop(index)
        logger.debug(f'{popped} popped from {self.id_} at index {index}')
        return popped

    def pop_and_update(self, instance, solution, indices: List[int]):
        """removes the vertex at the index position from the tour and resets all cost and schedules"""
        assert all(indices[i] <= indices[i + 1] for i in range(len(indices) - 1))  # assure that indices are sorted
        popped = []
        # iterate backwards because otherwise the 2nd index won't match after the 1st index has been removed
        for index in reversed(indices):
            popped.append(self._pop_no_update(index))
        self.update_sequences_and_schedules(instance, solution, indices[0])
        # the popped objects are in reverse order to the input indices, thus reverse again
        return reversed(popped)

    def update_sequences_and_schedules(self, instance, solution, index: int = 1):
        """update schedules from given index to end of routing sequence. index should be the same as the insertion
        or removal index. """
        for pos in range(index, len(self)):
            vertex: int = self.routing_sequence[pos]
            predecessor_vertex: int = self.routing_sequence[pos - 1]
            travel_dist = instance.distance([predecessor_vertex], [vertex])
            cumul_travel_dist = sum(self.travel_distance_sequence[:pos]) + travel_dist
            arrival = self.service_schedule[pos - 1] + ut.travel_time(travel_dist)
            cumul_load = sum(self.load_sequence[:pos]) + instance.load[vertex]
            # check the feasibility constraints
            if cumul_travel_dist >= instance.vehicles_max_travel_distance:
                message = f'{cumul_travel_dist} too long'
                logger.error(message)
                raise ut.ConstraintViolationError(message, message)
            if arrival > solution.tw_close[vertex]:
                message = f'{arrival} too late'
                logger.error(message)
                raise ut.ConstraintViolationError(message, message)
            if cumul_load > instance.vehicles_max_load:
                message = f'{cumul_load} to high'
                logger.error(message)
                raise ut.ConstraintViolationError(message, message)
            self._travel_distance_sequence[pos] = travel_dist
            self._travel_duration_sequence[pos] = ut.travel_time(travel_dist)
            self._arrival_schedule[pos] = arrival
            self._service_schedule[pos] = max(arrival, solution.tw_open[vertex])
            self._load_sequence[pos] = instance.load[vertex]
            self._revenue_sequence[pos] = instance.revenue[vertex]
        logger.debug(f'{self.id_} updated from index {index}')
        pass

    def reverse_section(self, instance, solution, i, j):
        """
        reverses a section of the route by connecting i->j and i+1 -> j+1.
        If reversal is infeasible, will raise InsertionError (and undoes the attempted reversal)

        Example: \n
        >> tour.sequence = [0, 1, 2, 3, 4, 0] \n
        >> tour.reverse_section(1, 4) \n
        >> print (tour.sequence) \n
        >> [0, 1, 4, 3, 2, 0]
        """
        for k in range(1, j - i):
            popped = self._pop_no_update(i + 1)
            self._insert_no_update(j - k + 1, popped)
        try:
            self.update_sequences_and_schedules(instance, solution, i)  # maybe the new routing sequence is infeasible
        except ut.InsertionError as e:
            self.reverse_section(instance, solution, i, j)  # undo all the reversal
            raise e

    def insertion_distance_delta(self, instance, indices: List[int], vertices: List[int]):
        """
        returns the distance surplus that is obtained by inserting the insertion_vertices at the insertion_positions.
        NOTE: Does NOT perform a feasibility check and does NOT actually insert the vertices!

        :param instance:
        :param indices: indices for insertion. provide for each vertex the index that it should be inserted at, CONSIDERING THAT PREVIOUSLY LISTED VERTICES UPDATE THE INDEX NUMBER OF ALL ITS (THEN) SUCCESSORS
        :param vertices:
        :return:
        """
        # raise NotImplementedError  # does not consider difference in consecutive vertices
        assert all(indices[i] <= indices[i + 1] for i in range(len(indices) - 1))  # assure that indices are sorted

        tmp_routing_sequence = list(self.routing_sequence)
        delta = 0
        for pos, vertex in zip(indices, vertices):
            predecessor_vertex = tmp_routing_sequence[pos - 1]
            successor_vertex = tmp_routing_sequence[pos]
            tmp_routing_sequence.insert(pos, vertex)
            delta += instance.distance([predecessor_vertex], [vertex]) + \
                     instance.distance([vertex], [successor_vertex]) - \
                     instance.distance([predecessor_vertex], [successor_vertex])

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
        # TODO avoid the copy if possible!
        tmp_routing_sequence = list(self.routing_sequence)
        for pos, vertex in zip(insertion_positions, insertion_vertices):
            tmp_routing_sequence.insert(pos, vertex)
        tour_length = sum(self.travel_distance_sequence[:insertion_positions[0]])
        arrival_time = self.arrival_schedule[insertion_positions[0] - 1]
        load = self.load_sequence[insertion_positions[0] - 1]

        # iterate over the temporary tour and check all constraints
        for pos in range(insertion_positions[0], len(tmp_routing_sequence)):
            vertex: int = tmp_routing_sequence[pos]
            # check precedence
            if vertex >= instance.num_carriers + instance.num_requests:
                precedence_vertex = vertex - instance.num_requests
                if precedence_vertex not in tmp_routing_sequence[:pos]:
                    return False
            predecessor_vertex: int = tmp_routing_sequence[pos - 1]
            travel_distance = instance.distance([predecessor_vertex], [vertex])
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
