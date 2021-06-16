import datetime as dt
import logging.config
from typing import List, Sequence

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
        self._routing_sequence: List[int] = []  # sequence of vertices
        self._travel_distance_sequence: List[float] = []  # sequence of arc distance costs
        self._travel_duration_sequence: List[dt.timedelta] = []  # sequence of arc duration costs
        self._load_sequence: List[float] = []  # the delta of the fill level of the vehicle (positive or negative)
        self._revenue_sequence: List[float] = []  # the collected revenue
        self._arrival_schedule: List[dt.datetime] = []  # arrival time at each stop
        self._service_schedule: List[dt.datetime] = []  # start of service time at each stop

        # details required for initialization procedure (and potentially for fast feasibility checks)
        self._wait_sequence: List[dt.timedelta] = []
        self._max_shift_sequence: List[dt.timedelta] = []

        # initialize depot to depot tour
        self.insert_and_update(instance, solution, [1], [depot_index])
        self.insert_and_update(instance, solution, [0], [depot_index])

    def __str__(self):
        return f'Tour ID:\t{self.id_}\n' \
               f'Sequence:\t{self.routing_sequence}\n' \
               f'Arrival:\t{[x.strftime("%d-%H:%M:%S") for x in self._arrival_schedule]}\n' \
               f'Service:\t{[x.strftime("%d-%H:%M:%S") for x in self._service_schedule]}\n' \
               f'Distance:\t{round(self.sum_travel_distance, 2)}\n' \
               f'Duration:\t{self.sum_travel_duration}\n' \
               f'Revenue:\t{self.sum_revenue}\n' \
               f'Profit:\t\t{self.sum_profit}\n'

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
    def travel_duration_sequence(self):
        return tuple(self._travel_duration_sequence)

    @property
    def sum_travel_duration(self) -> dt.timedelta:
        return sum(self._travel_duration_sequence, dt.timedelta())

    @property
    def load_sequence(self):
        return tuple(self._load_sequence)

    @property
    def sum_load(self):
        return sum(self.load_sequence)

    @property
    def revenue_sequence(self):
        return self._revenue_sequence

    @property
    def sum_revenue(self):
        return sum(self.revenue_sequence)

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

        # class method is just a wrapper around the function that is independent of the instance and solution class
        insert_no_update(
            routing_sequence=self._routing_sequence,
            travel_duration_sequence=self._travel_duration_sequence,
            travel_distance_sequence=self._travel_distance_sequence,
            arrival_schedule=self._arrival_schedule,
            service_schedule=self._service_schedule,
            load_sequence=self._load_sequence,
            revenue_sequence=self._revenue_sequence,
            wait_sequence=self._wait_sequence,
            max_shift_sequence=self._max_shift_sequence,
            index=index,
            vertex=vertex,
        )
        logger.debug(f'vertex {vertex} inserted into tour {self.id_} at index {index}')
        pass

    def insert_and_update(self, instance, solution, indices: Sequence[int], vertices: Sequence[int]):
        """
         inserts a vertex BEFORE the specified index and updates all sequences and schedules. Does not remove the
         request from the carrier's list of unrouted requests!

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
        popped = pop_no_update(
            routing_sequence=self._routing_sequence,
            travel_duration_sequence=self._travel_duration_sequence,
            travel_distance_sequence=self._travel_distance_sequence,
            arrival_schedule=self._arrival_schedule,
            service_schedule=self._service_schedule,
            load_sequence=self._load_sequence,
            revenue_sequence=self._revenue_sequence,
            wait_sequence=self._wait_sequence,
            max_shift_sequence=self._max_shift_sequence,
            index=index
        )
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

        # class method is just a wrapper around the function that is independent of the instance and solution class
        update_sequences_and_schedules(routing_sequence=self.routing_sequence,
                                       travel_distance_sequence=self._travel_distance_sequence,
                                       travel_duration_sequence=self._travel_duration_sequence,
                                       arrival_schedule=self._arrival_schedule,
                                       service_schedule=self._service_schedule,
                                       load_sequence=self._load_sequence,
                                       revenue_sequence=self._revenue_sequence,
                                       wait_sequence=self._wait_sequence,
                                       max_shift_sequence=self._max_shift_sequence,
                                       num_depots=instance.num_depots,
                                       num_requests=instance.num_requests,
                                       distance_matrix=instance._distance_matrix,
                                       vertex_load=instance.load,
                                       revenue=instance.revenue,
                                       service_duration=instance.service_duration,
                                       vehicles_max_travel_distance=instance.vehicles_max_travel_distance,
                                       vehicles_max_load=instance.vehicles_max_load,
                                       tw_open=solution.tw_open,
                                       tw_close=solution.tw_close,
                                       index=index)
        logger.debug(f'tour {self.id_} updated from index {index}')
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

    def insertion_distance_delta(self, instance, insertion_indices: List[int], vertices: List[int]):
        """
        returns the distance surplus that is obtained by inserting the insertion_vertices at the insertion_positions.
        NOTE: Does NOT perform a feasibility check and does NOT actually insert the vertices!

        """

        # class method is just a wrapper around the function that is independent of the instance and solution class
        return insertion_distance_delta(
            routing_sequence=self.routing_sequence,
            distance_matrix=instance._distance_matrix,
            insertion_indices=insertion_indices,
            vertices=vertices
        )

    def insertion_feasibility_check(self, instance,
                                    solution,
                                    insertion_positions: Sequence[int],
                                    insertion_vertices: Sequence[int]):
        """
        check Time Window, Maximum Tour Length and Maximum Vehicle Load constraints for the route IF the
        insertion_vertices were inserted at the insertion_positions. Insertion is not actually performed!

        :param instance:
        :param solution:
        :param insertion_positions:
        :param insertion_vertices:
        :return: True if all constraints are satisfied given the insertion; False otherwise
        """

        # wrapper around the version that is independent of instance and solution classes
        return insertion_feasibility_check(
            routing_sequence=self.routing_sequence,
            travel_distance_sequence=self.travel_distance_sequence,
            service_schedule=self.service_schedule,
            load_sequence=self.load_sequence,
            wait_sequence=self._wait_sequence,
            max_shift_sequence=self._max_shift_sequence,
            num_depots=instance.num_depots,
            num_requests=instance.num_requests,
            distance_matrix=instance._distance_matrix,
            vehicles_max_travel_distance=instance.vehicles_max_travel_distance,
            vertex_load=instance.load,
            service_duration=instance.service_duration,
            vehicles_max_load=instance.vehicles_max_load,
            tw_open=solution.tw_open,
            tw_close=solution.tw_close,
            insertion_positions=insertion_positions,
            insertion_vertices=insertion_vertices,

        )


# =====================================================================================================================
# stand-alone functions that are independent from the instance and solution classes but accept the raw data instead
# =====================================================================================================================

def feasibility_check(instance,
                      solution,
                      carrier: int,
                      original_tour: int,
                      new_routing_sequence: Sequence[int],
                      check_from_index: int
                      ):
    """
    Tests a new routing sequence for feasibility. That new sequence is a modification of the original old_tour and is
    usually obtained from a local search move.
    Will NOT check for max vehicle distance as it is assumed that no move that requires a feasibility check
    is detrimental for the total travel distance

    :param instance:
    :param solution:
    :param carrier:
    :param original_tour:
    :param new_routing_sequence:
    :param check_from_index:
    :return:
    """
    old_tour_: Tour = solution.carriers[carrier].tours[original_tour]
    return feasibility_check_raw(new_routing_sequence=new_routing_sequence,
                                 travel_distance_sequence=old_tour_.travel_distance_sequence,
                                 service_schedule=old_tour_.service_schedule,
                                 load_sequence=old_tour_.load_sequence,
                                 num_depots=instance.num_depots,
                                 num_requests=instance.num_requests,
                                 distance_matrix=instance._distance_matrix,
                                 vehicles_max_travel_distance=instance.vehicles_max_travel_distance,
                                 vertex_load=instance.load,
                                 service_duration=instance.service_duration,
                                 vehicles_max_load=instance.vehicles_max_load,
                                 tw_open=solution.tw_open,
                                 tw_close=solution.tw_close,
                                 check_from_index=check_from_index,
                                 )


def feasibility_check_raw(new_routing_sequence: Sequence[int],
                          travel_distance_sequence: Sequence[float],
                          service_schedule: Sequence[dt.datetime],
                          load_sequence: Sequence[float],
                          num_depots: int,
                          num_requests: int,
                          distance_matrix: Sequence[Sequence[float]],
                          vehicles_max_travel_distance: float,
                          vertex_load: Sequence[int],
                          service_duration: Sequence[dt.timedelta],
                          vehicles_max_load: float,
                          tw_open: Sequence[dt.datetime],
                          tw_close: Sequence[dt.datetime],
                          check_from_index: int,
                          ):
    """
    Tests a new routing sequence for feasibility. That new sequence is a modification of the original old_tour and is
    usually obtained from a local search move.

    :param new_routing_sequence: the NEW routing sequence which should be tested for feasibility
    :param travel_distance_sequence: the OLD sequence of travel distances
    :param service_schedule: the OLD list of start-of-service times
    :param load_sequence: the OLD sequence of load deltas
    :param num_depots: the total number of depots in the instance
    :param num_requests: the total number of requests in the instance
    :param distance_matrix:
    :param vehicles_max_travel_distance:
    :param vertex_load: the list of all vertices' load amount
    :param service_duration:
    :param vehicles_max_load:
    :param tw_open:
    :param tw_close:
    :param check_from_index: the index from which a feasibility check must start
    :return:
    """
    # initialize the values that must be checked to their correct value at the check_from_index
    total_travel_distance = sum(travel_distance_sequence[:check_from_index])
    service_time = service_schedule[check_from_index - 1]
    load = load_sequence[check_from_index - 1]

    # iterate over the temporary tour and check all constraints, starting from the check_from_index
    for pos in range(check_from_index, len(new_routing_sequence)):
        vertex: int = new_routing_sequence[pos]
        predecessor_vertex: int = new_routing_sequence[pos - 1]

        # check precedence, only required if it's a delivery vertex due to the order in which the tour is checked
        if num_depots + num_requests <= vertex < num_depots + 2 * num_requests:
            precedence_vertex = vertex - num_requests
            if precedence_vertex not in new_routing_sequence[:pos]:
                return False

        # max tour distance must not necessarily be checked since usually, only improving moves will be checked for
        # feasibility. but for now I'll keep it in
        travel_distance = distance_matrix[predecessor_vertex][vertex]
        total_travel_distance += travel_distance
        if total_travel_distance > vehicles_max_travel_distance:
            return False

        # check time windows
        arrival_time = service_time + service_duration[predecessor_vertex] + ut.travel_time(travel_distance)
        if arrival_time > tw_close[vertex]:
            return False

        # update the service time
        service_time = max(arrival_time, tw_open[vertex])

        # check max vehicle load
        load += vertex_load[vertex]
        if load > vehicles_max_load:
            return False

    return True


def insertion_feasibility_check(routing_sequence: Sequence[int],
                                travel_distance_sequence: Sequence[float],
                                service_schedule: Sequence[dt.datetime],
                                load_sequence: Sequence[float],
                                wait_sequence: Sequence[dt.timedelta],
                                # TODO with the help of wait_sequence and max_shift, insertion feasibility of a single vertex can be computed in constant time, I think
                                max_shift_sequence: Sequence[dt.timedelta],
                                num_depots: int,
                                num_requests: int,
                                distance_matrix: Sequence[Sequence[float]],
                                vehicles_max_travel_distance: float,
                                vertex_load: Sequence[int],
                                service_duration: Sequence[dt.timedelta],
                                vehicles_max_load: float,
                                tw_open: Sequence[dt.datetime],
                                tw_close: Sequence[dt.datetime],
                                insertion_positions: Sequence[int],
                                insertion_vertices: Sequence[int]):
    """
    check Time Window, Maximum Tour Length and Maximum Vehicle Load constraints for the route IF the
    insertion_vertices were inserted at the insertion_positions. Insertion is not actually performed!

    :return: True if all constraints are satisfied given the insertion; False otherwise
    """
    # make sure the insertion positions are ordered
    assert all(insertion_positions[i] < insertion_positions[i + 1] for i in range(len(insertion_positions) - 1))

    # create a temporary routing sequence to loop over that contains the new vertices
    tmp_routing_sequence = list(routing_sequence)
    for pos, vertex in zip(insertion_positions, insertion_vertices):
        tmp_routing_sequence.insert(pos, vertex)

    total_travel_dist = sum(travel_distance_sequence[:insertion_positions[0]])
    service_time = service_schedule[insertion_positions[0] - 1]
    load = load_sequence[insertion_positions[0] - 1]

    # iterate over the temporary tour and check all constraints
    for pos in range(insertion_positions[0], len(tmp_routing_sequence)):
        vertex: int = tmp_routing_sequence[pos]
        predecessor_vertex: int = tmp_routing_sequence[pos - 1]

        # check precedence if vertex is a delivery vertex
        if num_depots + num_requests <= vertex < num_depots + 2 * num_requests:
            precedence_vertex = vertex - num_requests
            if precedence_vertex not in tmp_routing_sequence[:pos]:
                return False

        # check max tour distance
        travel_distance = distance_matrix[predecessor_vertex][vertex]
        total_travel_dist += travel_distance
        if total_travel_dist > vehicles_max_travel_distance:
            return False

        # check time windows
        arrival_time = service_time + service_duration[predecessor_vertex] + ut.travel_time(travel_distance)
        if arrival_time > tw_close[vertex]:
            return False
        service_time = max(arrival_time, tw_open[vertex])

        # check max vehicle load
        load += vertex_load[vertex]
        if load > vehicles_max_load:
            return False
    return True


def insertion_distance_delta(routing_sequence: Sequence[int],
                             distance_matrix: Sequence[Sequence[float]],
                             insertion_indices: Sequence[int],
                             vertices: Sequence[int]
                             ):
    """

    :param routing_sequence:
    :param distance_matrix:
    :param insertion_indices:
    :param vertices:
    :return:
    """
    # assure that insertion_indices are sorted
    assert all(insertion_indices[i] <= insertion_indices[i + 1] for i in range(len(insertion_indices) - 1))

    tmp_routing_sequence = list(routing_sequence)
    delta = 0
    for pos, vertex in zip(insertion_indices, vertices):
        predecessor_vertex = tmp_routing_sequence[pos - 1]
        successor_vertex = tmp_routing_sequence[pos]
        tmp_routing_sequence.insert(pos, vertex)
        delta += distance_matrix[predecessor_vertex][vertex] + \
                 distance_matrix[vertex][successor_vertex] - \
                 distance_matrix[predecessor_vertex][successor_vertex]

    return delta


def update_sequences_and_schedules(routing_sequence: Sequence[int],
                                   travel_distance_sequence: List[float],
                                   travel_duration_sequence: List[dt.timedelta],
                                   arrival_schedule: List[dt.datetime],
                                   service_schedule: List[dt.datetime],
                                   load_sequence: List[float],
                                   revenue_sequence: List[float],
                                   wait_sequence: List[dt.timedelta],
                                   max_shift_sequence: List[dt.timedelta],
                                   num_depots: int,
                                   num_requests: int,
                                   distance_matrix: Sequence[Sequence[float]],
                                   vertex_load: Sequence[float],
                                   revenue: Sequence[float],
                                   service_duration: Sequence[dt.timedelta],
                                   vehicles_max_travel_distance: float,
                                   vehicles_max_load: float,
                                   tw_open: Sequence[dt.datetime],
                                   tw_close: Sequence[dt.datetime],
                                   index: int = 1):
    """update schedules from the given index to the end of routing sequence. index should be the same as the insertion
    or removal index."""

    total_travel_dist = sum(travel_distance_sequence[:index])
    load = sum(load_sequence[:index])

    for pos in range(index, len(routing_sequence)):
        vertex: int = routing_sequence[pos]
        predecessor_vertex: int = routing_sequence[pos - 1]

        # check precedence if the vertex is a delivery vertex
        if num_depots + num_requests <= vertex < num_depots + 2 * num_requests:
            precedence_vertex = vertex - num_requests
            if precedence_vertex not in routing_sequence[:pos]:
                message = f'Precedence Violation: vertex {precedence_vertex} must precede {vertex}!'
                logger.error(message)
                raise ut.ConstraintViolationError(message)

        # check max distance constraints
        travel_dist = distance_matrix[predecessor_vertex][vertex]
        total_travel_dist += travel_dist
        if total_travel_dist > vehicles_max_travel_distance:
            message = f'Distance {total_travel_dist} too long'
            logger.error(message)
            raise ut.ConstraintViolationError(message)

        # check tw constraints
        arrival_time = service_schedule[pos - 1] + service_duration[predecessor_vertex] + ut.travel_time(travel_dist)
        if arrival_time > tw_close[vertex]:
            vertex_type = "delivery" if num_depots + num_requests <= vertex < num_depots + 2 * num_requests else "pickup"
            message = f'arrival at vertex {vertex} ({vertex_type} ) in position {pos} at {arrival_time} too late, tw closes at {tw_close[vertex]}'
            logger.debug(message)
            raise ut.ConstraintViolationError(message, message)

        # check load constraints
        load += vertex_load[vertex]
        if load > vehicles_max_load:
            message = f'Load {load} to high'
            logger.error(message)
            raise ut.ConstraintViolationError(message, message)

        travel_distance_sequence[pos] = travel_dist
        travel_duration_sequence[pos] = ut.travel_time(travel_dist)
        arrival_schedule[pos] = arrival_time
        service_schedule[pos] = max(arrival_time, tw_open[vertex])
        load_sequence[pos] = vertex_load[vertex]
        revenue_sequence[pos] = revenue[vertex]

    # update wait_sequence and max_shift sequences. requires reverse traversal of the routing_sequence
    """
    Lu,Q., & Dessouky,M.M. (2006). A new insertion-based construction heuristic for solving the pickup 
    and delivery problem with time windows. European Journal of Operational Research, 175(2), 672–687.
    https://doi.org/10.1016/j.ejor.2005.05.012
    as well as
    Vansteenwegen,P., Souffriau,W., Vanden Berghe,G., & van Oudheusden,D. (2009). Iterated local search for the team
    orienteering problem with time windows. Computers & Operations Research, 36(12), 3281–3290.
    https://doi.org/10.1016/j.cor.2009.03.008
    """

    # start with the last vertex in the sequence
    pos = len(routing_sequence) - 1
    vertex = routing_sequence[pos]
    wait_sequence[pos] = max(dt.timedelta(0), arrival_schedule[pos] - tw_open[pos])
    max_shift_sequence[pos] = tw_close[vertex] - service_schedule[pos]

    # iterate over the remaining vertices; [note the range here!; taken from the paper: Lu,Q., & Dessouky,M.M. (2006)]
    for pos in range(len(routing_sequence) - 2, -1, -1):
        vertex: int = routing_sequence[pos]

        wait_sequence[pos] = max(dt.timedelta(0), arrival_schedule[pos] - tw_open[pos])
        max_shift_sequence[pos] = min(tw_close[vertex] - service_schedule[pos],
                                      max_shift_sequence[pos + 1] + wait_sequence[pos + 1])

    pass


def pop_and_update(indices: Sequence[int],
                   routing_sequence: List[int],
                   travel_distance_sequence: List[float],
                   travel_duration_sequence: List[dt.timedelta],
                   arrival_schedule: List[dt.datetime],
                   service_schedule: List[dt.datetime],
                   load_sequence: List[float],
                   revenue_sequence: List[float],
                   wait_sequence: List[dt.timedelta],
                   max_shift_sequence: List[dt.timedelta],
                   num_depots: int,
                   num_requests: int,
                   distance_matrix: Sequence[Sequence[float]],
                   vertex_load: Sequence[float],
                   revenue: Sequence[float],
                   service_duration: Sequence[dt.timedelta],
                   vehicles_max_travel_distance: float,
                   vehicles_max_load: float,
                   tw_open: Sequence[dt.datetime],
                   tw_close: Sequence[dt.datetime]):
    """removes the vertex at the index position from the tour and resets all cost and schedules"""
    assert all(indices[i] <= indices[i + 1] for i in range(len(indices) - 1))  # assure that indices are sorted
    popped = []
    # iterate backwards because otherwise the 2nd index won't match after the 1st index has been removed
    for index in reversed(indices):
        popped.append(pop_no_update(
            routing_sequence=routing_sequence,
            travel_duration_sequence=travel_duration_sequence,
            travel_distance_sequence=travel_distance_sequence,
            arrival_schedule=arrival_schedule,
            service_schedule=service_schedule,
            load_sequence=load_sequence,
            revenue_sequence=revenue_sequence,
            wait_sequence=wait_sequence,
            max_shift_sequence=max_shift_sequence,
            index=index))
    update_sequences_and_schedules(routing_sequence=routing_sequence,
                                   travel_distance_sequence=travel_distance_sequence,
                                   travel_duration_sequence=travel_duration_sequence,
                                   arrival_schedule=arrival_schedule,
                                   service_schedule=service_schedule,
                                   load_sequence=load_sequence,
                                   revenue_sequence=revenue_sequence,
                                   wait_sequence=wait_sequence,
                                   max_shift_sequence=max_shift_sequence,
                                   num_depots=num_depots,
                                   num_requests=num_requests,
                                   distance_matrix=distance_matrix,
                                   vertex_load=vertex_load,
                                   revenue=revenue,
                                   service_duration=service_duration,
                                   vehicles_max_travel_distance=vehicles_max_travel_distance,
                                   vehicles_max_load=vehicles_max_load,
                                   tw_open=tw_open,
                                   tw_close=tw_close,
                                   index=indices[0])
    # the popped objects are in reverse order to the input indices, thus reverse again
    return reversed(popped)


def pop_no_update(
        routing_sequence: List[int],
        travel_duration_sequence: List[dt.timedelta],
        travel_distance_sequence: List[float],
        arrival_schedule: List[dt.datetime],
        service_schedule: List[dt.datetime],
        load_sequence: List[float],
        revenue_sequence: List[float],
        wait_sequence: List[dt.timedelta],
        max_shift_sequence: List[dt.timedelta],
        index: int):
    """highly discouraged to use this as it does not ensure feasibility. only use for intermediate states that
    allow infeasible states such as for reversing a section! use pop_and_update instead"""
    popped = routing_sequence.pop(index)
    travel_duration_sequence.pop(index)
    travel_distance_sequence.pop(index)
    arrival_schedule.pop(index)
    service_schedule.pop(index)
    load_sequence.pop(index)
    revenue_sequence.pop(index)
    wait_sequence.pop(index)
    max_shift_sequence.pop(index)
    return popped


def insert_no_update(
        routing_sequence: List[int],
        travel_duration_sequence: List[dt.timedelta],
        travel_distance_sequence: List[float],
        arrival_schedule: List[dt.datetime],
        service_schedule: List[dt.datetime],
        load_sequence: List[float],
        revenue_sequence: List[float],
        wait_sequence: List[dt.timedelta],
        max_shift_sequence: List[dt.timedelta],
        index: int,
        vertex: int,
):
    routing_sequence.insert(index, vertex)
    travel_duration_sequence.insert(index, dt.timedelta(0))
    travel_distance_sequence.insert(index, 0)

    if len(routing_sequence) > 1:
        arrival_schedule.insert(index, None)
        service_schedule.insert(index, None)
        load_sequence.insert(index, None)
        revenue_sequence.insert(index, None)
        wait_sequence.insert(index, None)
        max_shift_sequence.insert(index, None)

    # only for initializing a tour with the first depot vertex
    else:
        arrival_schedule.insert(index, ut.START_TIME)
        service_schedule.insert(index, ut.START_TIME)
        load_sequence.insert(index, 0)
        revenue_sequence.insert(index, 0)
        wait_sequence.insert(index, dt.timedelta(0))
        max_shift_sequence.insert(index, ut.END_TIME - dt.datetime.min)
    pass


def insert_and_update(routing_sequence: List[int],
                      travel_distance_sequence: List[float],
                      travel_duration_sequence: List[dt.timedelta],
                      arrival_schedule: List[dt.datetime],
                      service_schedule: List[dt.datetime],
                      load_sequence: List[float],
                      revenue_sequence: List[float],
                      wait_sequence: List[dt.timedelta],
                      max_shift_sequence: List[dt.timedelta],
                      num_depots: int,
                      num_requests: int,
                      distance_matrix: Sequence[Sequence[float]],
                      vertex_load: Sequence[float],
                      revenue: Sequence[float],
                      service_duration: Sequence[dt.timedelta],
                      vehicles_max_travel_distance: float,
                      vehicles_max_load: float,
                      tw_open: Sequence[dt.datetime],
                      tw_close: Sequence[dt.datetime],
                      indices: Sequence[int],
                      vertices: Sequence[int]):
    assert all(indices[i] <= indices[i + 1] for i in range(len(indices) - 1))  # assure that indices are sorted

    for index, vertex in zip(indices, vertices):
        insert_no_update(
            routing_sequence=routing_sequence,
            travel_duration_sequence=travel_duration_sequence,
            travel_distance_sequence=travel_distance_sequence,
            arrival_schedule=arrival_schedule,
            service_schedule=service_schedule,
            load_sequence=load_sequence,
            revenue_sequence=revenue_sequence,
            wait_sequence=wait_sequence,
            max_shift_sequence=max_shift_sequence,
            index=index,
            vertex=vertex,
        )
    if len(routing_sequence) > 1:
        update_sequences_and_schedules(routing_sequence=routing_sequence,
                                       travel_distance_sequence=travel_distance_sequence,
                                       travel_duration_sequence=travel_duration_sequence,
                                       arrival_schedule=arrival_schedule,
                                       service_schedule=service_schedule,
                                       load_sequence=load_sequence,
                                       revenue_sequence=revenue_sequence,
                                       wait_sequence=wait_sequence,
                                       max_shift_sequence=max_shift_sequence,
                                       num_depots=num_depots,
                                       num_requests=num_requests,
                                       distance_matrix=distance_matrix,
                                       vertex_load=vertex_load,
                                       revenue=revenue,
                                       service_duration=service_duration,
                                       vehicles_max_travel_distance=vehicles_max_travel_distance,
                                       vehicles_max_load=vehicles_max_load,
                                       tw_open=tw_open,
                                       tw_close=tw_close,
                                       index=indices[0])
