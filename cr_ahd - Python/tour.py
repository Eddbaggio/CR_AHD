from copy import deepcopy
from typing import List
import matplotlib.pyplot as plt

import vertex as vx
from Optimizable import Optimizable
from solving.initializing_visitor import InitializingVisitor
from solving.local_search_visitor import FinalizingVisitor
from helper.utils import travel_time, opts, InsertionError


class Tour(Optimizable):
    def __init__(self, id_: str, depot: vx.DepotVertex, distance_matrix):
        self.id_ = id_
        self._distance_matrix = distance_matrix
        self._routing_sequence = []  # sequence of vertices
        self._cost_sequence = []
        self._cost = 0
        self.depot = depot
        # TODO: should this really be a sequence? What about the successor/adjacency-list? or at least a deque
        self.arrival_schedule = []  # arrival times  # TODO better make it a property that will be computed when requested. then, no compute_cost_and_schedules() is required any more
        self.service_schedule = []  # start of service times  # TODO better make it a property that will be computed when requested. then, no compute_cost_and_schedules() is required any more
        self._initializing_visitor: InitializingVisitor = None
        self._initialized = False
        self._finalizing_visitor: FinalizingVisitor = None
        self._finalized = False

        self.insert_and_update(0, depot)
        self.insert_and_update(1, depot)
        # self.compute_cost_and_schedules()

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
        self._cost += amount

    def decrease_cost(self, amount):
        assert amount <= self.cost, "Cannot have negative costs"
        self._cost -= amount

    def reset_cost_and_schedules(self):
        """
        Resets self.cost, self.arrival_schedule and self.service_schedule to None.
        """
        self.decrease_cost(self.cost)
        self.arrival_schedule = [None] * len(self)  # reset arrival times
        self.service_schedule = [None] * len(self)  # reset start of service times

    def insert_and_update(self, index: int, vertex: vx.BaseVertex):
        """
         inserts a vertex BEFORE the specified index and resets/deletes all cost and schedules.

        :param index: index before which the given vertex/request is to be inserted
        :param vertex: vertex to insert into the tour
        :return:
        """
        assert isinstance(vertex, vx.BaseVertex)
        self._routing_sequence.insert(index, vertex)
        vertex.routed = True
        self.update_cost_and_schedules_from(index)
        pass

    def pop_and_update(self, index: int):
        """removes the vertex at the index position from the tour and resets all cost and schedules"""
        removed = self._routing_sequence.pop(index)
        removed.routed = False
        self.update_cost_and_schedules_from(index)
        return removed

    def reverse_section(self, i, j):
        """
        reverses a section of the route from index i to index j-1.
        Example:
        tour.sequence = [0, 1, 2, 3, 4, 0]
        tour.reverse_section(1, 4)
        print (tour.sequence)
        >> [0, 3, 2, 1, 4, 0]
        """
        for k in range(1, j - i):
            popped = self.pop_and_update(i)
            self.insert_and_update(j - k, popped)

    def insertion_cost(self, i: vx.Vertex, j: vx.Vertex, u: vx.Vertex):
        """
        compute insertion cost for insertion of vertex u between vertices i and j using the distance matrix
        """
        dist_i_u = self.distance_matrix.loc[i.id_, u.id_]
        dist_u_j = self.distance_matrix.loc[u.id_, j.id_]
        dist_i_j = self.distance_matrix.loc[i.id_, j.id_]
        insertion_cost = dist_i_u + dist_u_j - dist_i_j
        return insertion_cost

    def update_cost_and_schedules_from(self, index: int = 1):
        """update schedules from given index to end of routing sequence. index should be the same as the insertion
        index. """
        for rho in range(index, len(self)):
            i: vx.Vertex = self.routing_sequence[rho - 1]
            j: vx.Vertex = self.routing_sequence[rho]
            dist = self.distance_matrix.loc[i.id_, j.id_]
            self._cost_sequence[rho] = dist
            arrival = self.service_schedule[rho - 1] + i.service_duration + travel_time(dist)
            assert arrival <= j.tw.l
            self.arrival_schedule[rho] = arrival
            self.service_schedule[rho] = min(arrival, j.tw.e)

    def compute_cost_and_schedules(self, start_time=opts['start_time'], ignore_tw=True, verbose=opts['verbose']):

        """
        Computes the total routing cost (distance-based) and the corresponding routing schedule (comprising sequence
        of vertex visits, arrival times and service start times). Based on an ASAP-principle (vehicle leaves as early as
        possible, serves as early as possible, No waiting strategies or similar things are applied)

        :param start_time:
        :param ignore_tw: Should time windows be ignored?
        :param verbose:
        :return:
        """
        raise NotImplementedError
        self.decrease_cost(self.cost)
        self.arrival_schedule[0] = start_time
        self.service_schedule[0] = start_time
        for rho in range(1, len(self)):
            i: vx.Vertex = self.routing_sequence[rho - 1]
            j: vx.Vertex = self.routing_sequence[rho]
            dist = self.distance_matrix.loc[i.id_, j.id_]
            self.increase_cost(dist)  # sum up the total routing cost
            planned_arrival = self.service_schedule[rho - 1] + i.service_duration + travel_time(dist)
            if verbose > 2:
                print(f'Planned arrival at {j}: {planned_arrival}')
            if not ignore_tw:
                assert planned_arrival <= j.tw.l  # assert that arrival happens before time window closes
            self.arrival_schedule[rho] = planned_arrival  # set the arrival time
            # if arrival is later than time window opening, choose service time = arrival time
            if planned_arrival >= j.tw.e:
                self.service_schedule[rho] = planned_arrival
            # else if arrival is later than time window opening, choose service time =  tw opening
            else:
                self.service_schedule[rho] = j.tw.e
        pass

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

    def find_cheapest_feasible_insertion(self, u: vx.Vertex, verbose=opts['verbose']):
        """
        :return: Tuple (position, cost) of the best insertion position index and the associated (lowest) cost
        """
        assert u.routed == False, f'Trying to insert an already routed vertex! {u}'
        if verbose > 2:
            print(f'\n= Cheapest insertion of {u.id_} into {self.id_}')
            print(self)
        min_insertion_cost = float('inf')
        i_best = None
        j_best = None

        # test all insertion positions
        for rho in range(1, len(self)):
            i: vx.Vertex = self.routing_sequence[rho - 1]
            j: vx.Vertex = self.routing_sequence[rho]

            # trivial feasibility check
            if i.tw.e < u.tw.l and u.tw.e < j.tw.l:
                # compute insertion cost
                insertion_cost = self.insertion_cost(i, j, u)

                if verbose > 2:
                    print(f'Between {i.id_} and {j.id_}: {insertion_cost}')

                if insertion_cost < min_insertion_cost:
                    # check feasibility
                    self.insert_and_update(index=rho, vertex=u)
                    if self.is_feasible():
                        # update best known insertion position
                        min_insertion_cost = insertion_cost
                        i_best = i
                        j_best = j
                        insertion_position = rho
                    self.pop_and_update(rho)
        # return the best found position and cost or raise an error if no feasible position was found
        if i_best:
            if verbose > 2:
                print(f'== Best: between {i_best.id_} and {j_best.id_}: {min_insertion_cost}')
            return insertion_position, min_insertion_cost
        else:
            raise InsertionError('', 'No feasible insertion position found')

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
        self.compute_cost_and_schedules(ignore_tw=True)  # TODO Why do I ignore TW here?
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
        self.compute_cost_and_schedules()
        for rho in range(1, len(self)):
            i = self.routing_sequence[rho - 1]
            j = self.routing_sequence[rho]

            # trivial feasibility check
            if i.tw.e < u.tw.l and u.tw.e < j.tw.l:
                # proper feasibility check
                # TODO: check Solomon (1987) for an efficient and sufficient feasibility check
                if self.is_feasible():
                    # compute c1(=best feasible insertion cost) and c2(=best request) and update their best values
                    c1 = self.c1(i_index=rho - 1,
                                 u=u,
                                 j_index=rho,
                                 alpha_1=opts['alpha_1'],
                                 mu=opts['mu'],
                                 )
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
        :return: List of artists to be drawn. Use for animated plotting
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
    from helper.utils import make_dist_matrix

    requests = [vx.Vertex(f'{i}', i, i, 0, 0, 10, 0) for i in range(3)]
    new_request = vx.Vertex('new', 99, 99, 0, 0, 10, 0)
    distance_matrix = make_dist_matrix([*requests, new_request])
    tour = Tour('tour', vx.DepotVertex('depot', 0, 0), distance_matrix)
    for r in requests:
        tour.insert_and_update(1, r)
    print(tour)

    new_tour = deepcopy(tour)
    print(new_request)
    new_tour.insert_and_update(1, new_request)
    print(new_request)

    print('End')
