from copy import deepcopy
from typing import List
import matplotlib.pyplot as plt

from src.cr_ahd.core_module.optimizable import Optimizable
import src.cr_ahd.core_module.vertex as vx
from src.cr_ahd.solving_module.tour_initialization import TourInitializationBehavior
from src.cr_ahd.solving_module.tour_improvement import TourImprovementBehavior
from src.cr_ahd.utility_module.utils import travel_time, opts, InsertionError


class Tour(Optimizable):
    def __init__(self, id_: str, depot: vx.DepotVertex, distance_matrix):
        self.id_ = id_
        self.depot = depot
        self._distance_matrix = distance_matrix
        self._routing_sequence = []  # sequence of vertices
        self._cost_sequence = []  # sequence of arc costs
        self._cost = 0
        self.arrival_schedule = []  # arrival times
        self.service_schedule = []  # start of service times
        # self._initializing_visitor: InitializingVisitor = None
        # self._initialized = False
        # self._finalizing_visitor: FinalizingVisitor = None
        # self._finalized = False

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
                arrival_schedule.append(round(self.arrival_schedule[i], 2))
                service_schedule.append(round(self.service_schedule[i], 2))
        if self.cost is None:
            cost = None
        else:
            cost = round(self.cost)
        return f'ID:\t\t\t{self.id_}\nSequence:\t{sequence}\nArrival:\t{arrival_schedule}\nService:\t{service_schedule}\nCost:\t\t{cost}'

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
        return self.revenue - self.cost

    @property  # why exactly is this a property and not just a member attribute?
    def distance_matrix(self):
        return self._distance_matrix

    @distance_matrix.setter
    def distance_matrix(self, distance_matrix):
        self._distance_matrix = distance_matrix

    '''
    @property
    def finalizing_visitor(self):
        """the finalizer local search optimization, such as 2opt or 3opt"""
        return self._finalizing_visitor

    @finalizing_visitor.setter
    def finalizing_visitor(self, visitor):
        """Setter for the local search algorithm that can be used to finalize the results"""
        # assert (not self._finalized), f"Instance has been finalized with " \
        #                               f"visitor {self._finalizing_visitor.__class__.__name__} already!"
        self._finalizing_visitor = visitor
    '''

    @property
    def cost(self):
        return sum(self._cost_sequence)

    def _insert_no_update(self, index: int, vertex: vx.BaseVertex):
        """highly discouraged to use this as it does not ensure feasibility of the route! use insert_and_update instead.
        use this only if infeasible route is acceptable, as e.g. reversing a section where intermediate states of the
        reversal may be infeasible"""
        assert isinstance(vertex, vx.BaseVertex)
        self._routing_sequence.insert(index, vertex)
        vertex.routed = [self, index]
        if len(self) <= 1:
            self._cost_sequence.insert(index, 0)
            self.arrival_schedule.insert(index, opts['start_time'])
            self.service_schedule.insert(index, opts['start_time'])
        else:
            self._cost_sequence.insert(index, None)
            self.arrival_schedule.insert(index, None)
            self.service_schedule.insert(index, None)

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
        except InsertionError as e:
            self.pop_and_update(index)
            raise e
        pass

    def _pop_no_update(self, index: int):
        """highly discouraged to use this as it does not ensure feasibility. only use for intermediate states that
        allow infeasible states such as for reversing a section! use pop_and_update instead"""
        popped = self._routing_sequence.pop(index)
        self._cost_sequence.pop(index)
        self.arrival_schedule.pop(index)
        self.service_schedule.pop(index)
        popped.routed = False
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
            i.routed[1] = rho-1  # update the stored index position
            dist = self.distance_matrix.loc[i.id_, j.id_]
            self._cost_sequence[rho] = dist
            arrival = self.service_schedule[rho - 1] + i.service_duration + travel_time(dist)
            try:
                assert arrival <= j.tw.l
                self.arrival_schedule[rho] = arrival
                self.service_schedule[rho] = max(arrival, j.tw.e)
            except AssertionError:
                raise InsertionError(f'{arrival} <= {j.tw.l}', f'cannot arrive at {j} after time window has closed')
            # TODO "You shouldn't throw exceptions for things that happen all the time. Then they'd be "ordinaries"."
            #  https://softwareengineering.stackexchange.com/questions/139171/check-first-vs-exception-handling

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
        except InsertionError as e:
            self.reverse_section(i, j)  # undo all the reversal
            raise e

    def insertion_cost_no_fc(self, i: vx.Vertex, j: vx.Vertex, u: vx.Vertex):  # TODO what about Time Windows?
        """
        compute insertion cost for insertion of vertex u between vertices i and j using the distance matrix. Does NOT
        consider time window restrictions, i.e. no feasibility check is done!
        """
        dist_i_u = self.distance_matrix.loc[i.id_, u.id_]
        dist_u_j = self.distance_matrix.loc[u.id_, j.id_]
        dist_i_j = self.distance_matrix.loc[i.id_, j.id_]
        insertion_cost = dist_i_u + dist_u_j - dist_i_j
        return insertion_cost

    '''
    def finalize(self, visitor):
        """apply visitor's local search procedure to improve the result after the routing itself has been done"""
        assert (not self._finalized), f'Instance has been finalized with {self._finalizing_visitor} already'
        self._finalizing_visitor = visitor
        visitor.finalize_tour(self)
        pass
    '''

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


if __name__ == '__main__':
    pass
