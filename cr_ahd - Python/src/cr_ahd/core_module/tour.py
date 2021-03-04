from copy import deepcopy
from typing import List
import matplotlib.pyplot as plt

from src.cr_ahd.core_module.Optimizable import Optimizable
import src.cr_ahd.core_module.vertex as vx
from src.cr_ahd.solving_module.initializing_visitor import InitializingVisitor
from src.cr_ahd.solving_module.local_search_visitor import FinalizingVisitor
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
        self._initializing_visitor: InitializingVisitor = None
        self._initialized = False
        self._finalizing_visitor: FinalizingVisitor = None
        self._finalized = False

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

    @property
    def finalizing_visitor(self):
        """the finalizer local search optimization, such as 2opt or 3opt"""
        return self._finalizing_visitor

    @finalizing_visitor.setter
    def finalizing_visitor(self, visitor):
        """Setter for the local search algorithm that can be used to finalize the results"""
        assert (not self._finalized), f"Instance has been finalized with " \
                                      f"visitor {self._finalizing_visitor.__class__.__name__} already!"
        self._finalizing_visitor = visitor

    @property
    def cost(self):
        return sum(self._cost_sequence)

    def increase_cost(self, amount):
        raise NotImplementedError  # has been replaced with the cost_sequence
        self._cost += amount

    def decrease_cost(self, amount):
        raise NotImplementedError  # has been replaced with the cost_sequence
        assert amount <= self.cost, "Cannot have negative costs"
        self._cost -= amount

    def reset_cost_and_schedules(self):
        """
        Resets self.cost, self.arrival_schedule and self.service_schedule to None.
        """
        self.decrease_cost(self.cost)
        self.arrival_schedule = [None] * len(self)  # reset arrival times
        self.service_schedule = [None] * len(self)  # reset start of service times

    def _insert_no_update(self, index: int, vertex: vx.BaseVertex):
        """highly discouraged to use this as it does not ensure feasibility of the route! use insert_and_update instead.
        use this only if infeasible route is acceptable, as e.g. reversing a section where intermediate states of the
        reversal may be infeasible"""
        assert isinstance(vertex, vx.BaseVertex)
        self._routing_sequence.insert(index, vertex)
        vertex.routed = True
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

    def _pop_no_update(self, index:int):
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

    def is_feasible(self, start_time=opts['start_time'], verbose=opts['verbose']) -> bool:
        """

        :param start_time:
        :param verbose:
        :return:
        """
        if verbose > 2:
            print(f'== Feasibility Check')
            print(self)
        service_schedule = self.service_schedule.copy()
        service_schedule[0] = start_time
        for rho in range(1, len(self)):
            i: vx.Vertex = self.routing_sequence[rho - 1]
            j: vx.Vertex = self.routing_sequence[rho]
            dist = self.distance_matrix.loc[i.id_, j.id_]
            if verbose > 3:
                print(f'iteration {rho}; service_schedule: {service_schedule}')
            earliest_arrival = service_schedule[rho - 1] + i.service_duration + travel_time(dist)
            if earliest_arrival > j.tw.l:
                if verbose > 2:
                    # print(f'From {i} to {j}')
                    # print(f'Predecessor start of service : {service_schedule[rho - 1]}')
                    # print(f'Predecessor service time : {i.service_duration}')
                    # print(f'Distance : {dist}')
                    # print(f'Travel time: {travel_time(dist)}')
                    print(f'Infeasible! {round(earliest_arrival, 2)} > {j.id_}.tw.l: {j.tw.l}')
                    # print()
                return False
            else:
                service_schedule[rho] = max(j.tw.e, earliest_arrival)
        return True

    def c1(self,
           i_index: int,
           u: vx.Vertex,
           j_index: int,
           alpha_1: float,
           mu: float,
           ) -> float:
        """
        c1 criterion of Solomon's I1 insertion heuristic: "best feasible insertion cost"
        Does NOT include a feasibility check. Following the
        description by (Bräysy, Olli; Gendreau, Michel (2005): Vehicle Routing Problem with Time Windows,
        Part I: Route Construction and Local Search Algorithms. In Transportation Science 39 (1), pp. 104–118. DOI:
        10.1287/trsc.1030.0056.)
        """
        alpha_2 = 1 - alpha_1

        i = self.routing_sequence[i_index]
        j = self.routing_sequence[j_index]

        # compute c11
        c11 = self._c11(i, u, j, mu)

        # compute c12
        c12 = self._c12(j_index, u)

        # compute and return c1
        return alpha_1 * c11 + alpha_2 * c12

    def _c11(self, i, u, j, mu):
        """weighted insertion cost"""
        c11 = self.distance_matrix.loc[i.id_, u.id_] + self.distance_matrix.loc[u.id_, j.id_] - mu * \
              self.distance_matrix.loc[i.id_, j.id_]
        return c11

    def _c12(self, j_index, u):
        """how much will the arrival time of vertex at index j be pushed back?"""
        service_start_j = self.service_schedule[j_index]
        self.insert_and_update(index=j_index, vertex=u)
        service_start_j_new = self.service_schedule[j_index + 1]
        c12 = service_start_j_new - service_start_j
        self.pop_and_update(j_index)
        return c12

    def c2(self,
           u: vx.Vertex,
           c1: float,
           lambda_: float = opts['lambda'],
           ):
        """
        c2 criterion of Solomon's I1 insertion heuristic: "find the best customer/request"
        Does NOT include a feasibility check Following the
        description by (Bräysy, Olli; Gendreau, Michel (2005): Vehicle Routing Problem with Time Windows,
        Part I: Route Construction and Local Search Algorithms. In Transportation Science 39 (1), pp. 104–118. DOI:
        10.1287/trsc.1030.0056.)
        """
        return lambda_ * self.distance_matrix.loc[self.depot.id_, u.id_] - c1

    def find_best_feasible_I1_insertion(self, u: vx.Vertex, verbose=opts['verbose']):
        """
        returns float('-inf') if no feasible insertion position was found
        :param u:
        :param verbose:
        :return:
        """
        rho_best = None
        max_c2 = float('-inf')
        for rho in range(1, len(self)):
            i = self.routing_sequence[rho - 1]
            j = self.routing_sequence[rho]

            # trivial feasibility check
            if i.tw.e < u.tw.l and u.tw.e < j.tw.l:
                # TODO: check Solomon (1987) for an efficient and sufficient feasibility check
                # compute c1(=best feasible insertion cost) and c2(=best request) and update their best values
                try:
                    c1 = self.c1(i_index=rho - 1,
                                 u=u,
                                 j_index=rho,
                                 alpha_1=opts['alpha_1'],
                                 mu=opts['mu'],
                                 )
                except InsertionError:
                    continue
                if verbose > 1:
                    print(f'c1({u.id_}->{self.id_}): {c1}')
                c2 = self.c2(u=u, lambda_=opts['lambda'], c1=c1)
                if verbose > 1:
                    print(f'c2({u.id_}->{self.id_}): {c2}')
                if c2 > max_c2:
                    max_c2 = c2
                    rho_best = rho
        return rho_best, max_c2

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
