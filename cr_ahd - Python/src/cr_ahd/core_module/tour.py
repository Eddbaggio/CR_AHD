import datetime as dt
import logging.config
from typing import List

import matplotlib.pyplot as plt

import src.cr_ahd.core_module.vertex as vx
from src.cr_ahd.core_module.optimizable import Optimizable
import src.cr_ahd.utility_module.utils as ut

logger = logging.getLogger(__name__)


class Tour(Optimizable):
    def __init__(self, id_: str, depot: vx.DepotVertex, distance_matrix):
        self.id_ = id_
        self.depot = depot
        self._distance_matrix = distance_matrix
        self._routing_sequence = []  # sequence of vertices
        self._travel_dist_sequence: List[float] = []  # sequence of arc distance costs
        self._travel_duration_sequence: List[dt.timedelta] = []  # sequence of arc duration costs
        self._sum_travel_times = 0
        self.arrival_schedule: List[dt.datetime] = []  # arrival times
        self.service_schedule: List[dt.datetime] = []  # start of service times

        self.insert_and_update(0, depot)
        self.insert_and_update(1, depot)

    def __str__(self):
        sequence = [vertex.id_ for vertex in self.routing_sequence]
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
        duration = self.sum_travel_duration
        return f'ID:\t\t\t{self.id_}\nSequence:\t{sequence}\nArrival:\t{arrival_schedule}\n' \
               f'Service:\t{service_schedule}\nDistance:\t\t{distance}\nDuration:\t\t{duration} '

    def __len__(self):
        return len(self.routing_sequence)

    @property
    def routing_sequence(self):
        """immutable sequence of routed vertices. can only be modified by inserting"""
        return tuple(self._routing_sequence)

    @property
    def revenue(self):
        return sum([request.demand for request in self.routing_sequence])

    @property
    def profit(self):
        return self.revenue - self.sum_travel_duration

    @property  # why exactly is this a property and not just a member attribute?
    def distance_matrix(self):
        return self._distance_matrix

    @distance_matrix.setter
    def distance_matrix(self, distance_matrix):
        self._distance_matrix = distance_matrix

    @property
    def sum_travel_duration(self) -> dt.timedelta:
        return sum(self._travel_duration_sequence, dt.timedelta())

    @property
    def sum_travel_distance(self) -> float:
        return sum(self._travel_dist_sequence)

    def _insert_no_update(self, index: int, vertex: vx.BaseVertex):
        """highly discouraged to use this as it does not ensure feasibility of the route! use insert_and_update instead.
        use this only if infeasible route is acceptable, as e.g. reversing a section where intermediate states of the
        reversal may be infeasible"""
        assert isinstance(vertex, vx.BaseVertex)
        self._routing_sequence.insert(index, vertex)
        vertex.routed = [self, index]
        if len(self) <= 1:
            self._travel_duration_sequence.insert(index, dt.timedelta(0))
            self._travel_dist_sequence.insert(index, 0)
            self.arrival_schedule.insert(index, ut.opts['start_time'])
            self.service_schedule.insert(index, ut.opts['start_time'])
        else:
            self._travel_duration_sequence.insert(index, dt.timedelta(0))
            self._travel_dist_sequence.insert(index, 0)
            self.arrival_schedule.insert(index, None)
            self.service_schedule.insert(index, None)
        logger.debug(f'{vertex.id_} inserted into {self.id_} at index {index}')
        pass

    def insert_and_update(self, index: int, vertex: vx.BaseVertex):
        """
         inserts a vertex BEFORE the specified index and resets/deletes all cost and schedules. If insertion is
         infeasible due to time window constraints, it will undo the insertion and raise InsertionError

        :param index: index before which the given vertex/request is to be inserted
        :param vertex: vertex to insert into the tour
        """
        try:
            self._insert_no_update(index, vertex)
            if len(self) > 1:
                self.update_cost_and_schedules_from(index)
        except ut.InsertionError as e:
            self.pop_and_update(index)
            raise e
        pass

    def _pop_no_update(self, index: int):
        """highly discouraged to use this as it does not ensure feasibility. only use for intermediate states that
        allow infeasible states such as for reversing a section! use pop_and_update instead"""
        popped = self._routing_sequence.pop(index)
        self._travel_duration_sequence.pop(index)
        self._travel_dist_sequence.pop(index)
        self.arrival_schedule.pop(index)
        self.service_schedule.pop(index)
        popped.routed = False
        logger.debug(f'{popped.id_} popped from {self.id_} at index {index}')
        return popped

    def pop_and_update(self, index: int):
        """removes the vertex at the index position from the tour and resets all cost and schedules"""
        popped = self._pop_no_update(index)
        self.update_cost_and_schedules_from(index)
        return popped

    def update_cost_and_schedules_from(self, index: int = 1):
        """update schedules from given index to end of routing sequence. index should be the same as the insertion
        or removal index. """
        for rho in range(index, len(self)):
            i: vx.Vertex = self.routing_sequence[rho - 1]
            j: vx.Vertex = self.routing_sequence[rho]
            i.routed[1] = rho - 1  # update the stored index position
            dist = self.distance_matrix.loc[i.id_, j.id_]
            self._travel_dist_sequence[rho] = dist
            self._travel_duration_sequence[rho] = ut.travel_time(dist)
            arrival = self.service_schedule[rho - 1] + i.service_duration + ut.travel_time(dist)
            try:
                assert arrival <= j.tw.l, f'Arrival at {arrival} is too late for {j}'
                self.arrival_schedule[rho] = arrival
                self.service_schedule[rho] = max(arrival, j.tw.e)
            except AssertionError:
                logger.debug(f'{self.id_} update from index {index} failed: arrival:{arrival} > j.tw.l.:{j.tw.l}')
                raise ut.InsertionError(f'{arrival} <= {j.tw.l}', f'cannot arrive at {j} after time window has closed')
            # TODO "You shouldn't throw exceptions for things that happen all the time. Then they'd be "ordinaries"."
            #  https://softwareengineering.stackexchange.com/questions/139171/check-first-vs-exception-handling
        logger.debug(f'{self.id_} updated from index {index}')
        pass

    def reverse_section(self, i, j):
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
            self.update_cost_and_schedules_from(i)  # maybe the new routing sequence is infeasible
        except ut.InsertionError as e:
            self.reverse_section(i, j)  # undo all the reversal
            raise e

    def insertion_distance_cost_no_feasibility_check(self, i: vx.Vertex, j: vx.Vertex,
                                                     u: vx.Vertex):  # TODO what about Time Windows?
        """
        compute distance cost for insertion of vertex u between vertices i and j using the distance matrix. Does NOT
        consider time window restrictions, i.e. no feasibility check is done!
        """
        dist_i_u = self.distance_matrix.loc[i.id_, u.id_]
        dist_u_j = self.distance_matrix.loc[u.id_, j.id_]
        dist_i_j = self.distance_matrix.loc[i.id_, j.id_]
        insertion_cost = dist_i_u + dist_u_j - dist_i_j
        return insertion_cost

    def plot(self,
             plot_depot: bool = True,
             annotate: bool = True,
             alpha: float = 1,
             color: str = 'black') -> List:

        """
        :return: List of artists to be drawn. Use for ANIMATED plotting
        """
        artists = []  # list of artists to be plotted

        # plot depot
        if plot_depot:
            depot, = plt.plot(*self.depot.coords,
                              marker='s',
                              markersize='9',
                              mfc=color,
                              mec='black',
                              mew=1,
                              alpha=alpha,
                              label=self.depot.id_,
                              linestyle='')
            artists.append(depot)

        # plot requests locations
        x = [r.coords.x for r in self.routing_sequence[1:-1]]
        y = [r.coords.y for r in self.routing_sequence[1:-1]]
        requests, = plt.plot(x, y, marker='o', c=color, alpha=alpha, label=self.id_, linestyle='')
        artists.append(requests)

        # plot arrows and annotations
        for i in range(1, len(self)):
            start = self.routing_sequence[i - 1]
            end = self.routing_sequence[i]
            arrows = plt.annotate('',
                                  xy=(end.coords.x, end.coords.y),
                                  xytext=(start.coords.x, start.coords.y),
                                  c=color,
                                  arrowprops=dict(arrowstyle="-|>",
                                                  alpha=alpha,
                                                  color=requests.get_color(),
                                                  ),
                                  )
            artists.append(arrows)

            if annotate:
                annotations = plt.annotate(f'{end.id_}',
                                           xy=(end.coords.x, end.coords.y + 1),
                                           alpha=alpha
                                           )
                artists.append(annotations)
        return artists
