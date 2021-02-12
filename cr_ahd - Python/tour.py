from copy import deepcopy
from typing import List

import matplotlib.pyplot as plt

import vertex as vx
from Optimizable import Optimizable
from solution_visitors.initializing_visitor import InitializingVisitor
from solution_visitors.local_search_visitor import FinalizingVisitor
from utils import travel_time, opts, InsertionError


class Tour(Optimizable):
    def __init__(self, id_: str, sequence: List[vx.Vertex], distance_matrix):
        self.id_ = id_
        self._distance_matrix = distance_matrix
        self.sequence = sequence  # sequence of vertices
        self.depot = sequence[0]
        # TODO: should this really be a sequence? What about the successor/adjacency-list?
        self.arrival_schedule = [None] * len(sequence)  # arrival times
        self.service_schedule = [None] * len(sequence)  # start of service times
        self._cost = 0  # TODO better make this a property? -> Yes! + have increase_cost, reduce_cost methods but no setter
        self._initializing_visitor: InitializingVisitor = None
        self._initialized = False
        self._finalizing_visitor: FinalizingVisitor = None
        self._finalized = False

    def __str__(self):
        sequence = [vertex.id_ for vertex in self.sequence]
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
        return len(self.sequence)

    @property
    def revenue(self):
        return sum([r.demand for r in self.sequence])

    @property
    def profit(self):
        return self.revenue - self.cost

    @property
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
        assert (
            not self._finalized), f"Instance has been finalized with visitor {self._finalizing_visitor.__class__.__name__} already!"
        self._finalizing_visitor = visitor

    @property
    def cost(self):
        return self._cost

    def increase_cost(self, amount):
        self._cost += amount

    def decrease_cost(self, amount):
        assert amount <= self.cost, "Cannot have negative costs"
        self._cost -= amount

    # def copy(self):
    #     tour = Tour(self.sequence)
    #     tour.arrival_schedule = self.arrival_schedule
    #     tour.service_schedule = self.service_schedule
    #     tour.cost = self.cost
    #     return tour

    def reset_solution(self):
        """Replace existing sequence and schedules by depot-depot schedules and set cost to 0"""
        self.decrease_cost(self.cost)
        self.sequence = [self.depot, self.depot]
        self.arrival_schedule = None
        self.service_schedule = None

    def reset_cost_and_schedules(self):
        """
        Resets self.cost, self.arrival_schedule and self.service_schedule to None.
        """
        self.decrease_cost(self.cost)
        self.arrival_schedule = [None] * len(self)  # reset arrival times
        self.service_schedule = [None] * len(self)  # reset start of service times

    def copy_cost_and_schedules(self, other):
        """copy sequence, arrival schedule, service schedule and cost from other to self"""
        # TODO maybe i should rather attempt operator overloading? self = other
        self.sequence = other.sequence
        self.arrival_schedule = other.arrival_schedule
        self.service_schedule = other.service_schedule
        self.decrease_cost(self.cost)
        self.increase_cost(other.cost)

    def insert_and_reset(self, index: int, vertex: vx.Vertex):
        """
         inserts a vertex before the specified index and resets/deletes all cost and schedules.

        :param index: index before which the given vertex/request is to be inserted
        :param vertex: vertex to insert into the tour
        :return:
        """
        assert type(vertex) == vx.Vertex
        self.sequence.insert(index, vertex)
        self.reset_cost_and_schedules()
        pass

    # TODO: custom versions of the list methods, e.g. insert method that automatically updates the schedules?!
    def _insert_and_update_schedules(self):
        pass

    # def is_insertion_feasible(self, index: int, vertex: vx.Vertex, dist_matrix, verbose=opts['verbose']) -> bool:
    #     i = self.sequence[index - 1]
    #     j = self.sequence[index]
    #     # trivial feasibility check
    #     if i.tw.e < vertex.tw.l and vertex.tw.e < j.tw.l:
    #         # check feasibility
    #         temp_tour = deepcopy(self)
    #         temp_tour.insert_and_reset_schedules(index=index, vertex=vertex)
    #         if temp_tour.is_feasible(dist_matrix=dist_matrix):
    #             return True
    #         else:
    #             return False
    #     else:
    #         return False

    def compute_cost_and_schedules(self, start_time=opts['start_time'], ignore_tw=True,
                                   verbose=opts['verbose']):

        """
        Computes the total routing cost (distance-based) and the corresponding routing schedule (comprising sequence
        of vertex visits, arrival times and service start times). Based on an ASAP-principle (vehicle leaves as early as
        possible, serves as early as possible, No waiting strategies or similar things are applied)

        :param dist_matrix: The distance matrix to use
        :param start_time:
        :param ignore_tw: Should time windows be ignored?
        :param verbose:
        :return:
        """
        self.decrease_cost(self.cost)
        self.arrival_schedule[0] = start_time
        self.service_schedule[0] = start_time
        for rho in range(1, len(self)):
            i: vx.Vertex = self.sequence[rho - 1]
            j: vx.Vertex = self.sequence[rho]
            dist = self.distance_matrix.loc[i.id_, j.id_]
            self.increase_cost(dist)  # sum up the total routing cost
            planned_arrival = self.service_schedule[rho - 1] + i.service_duration + travel_time(dist)
            if verbose > 2:
                print(f'Planned arrival at {j}: {planned_arrival}')
            if not ignore_tw:
                assert planned_arrival <= j.tw.l  # assert that arrival happens before time window closes
            self.arrival_schedule[rho] = planned_arrival  # set the arrival time
            if planned_arrival >= j.tw.e:
                self.service_schedule[
                    rho] = planned_arrival  # if arrival is later than time window opening, choose service time = arrival time
            else:
                self.service_schedule[
                    rho] = j.tw.e  # if arrival is later than time window opening, chooseservie time =  tw opening
        pass

    def is_feasible(self, start_time=opts['start_time'], verbose=opts['verbose']) -> bool:
        """

        :param dist_matrix:
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
            i: vx.Vertex = self.sequence[rho - 1]
            j: vx.Vertex = self.sequence[rho]
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
        if verbose > 2:
            print(f'\n= Cheapest insertion of {u.id_} into {self.id_}')
            print(self)
        min_insertion_cost = float('inf')
        i_best = None
        j_best = None

        # test all insertion positions
        for rho in range(1, len(self)):

            i: vx.Vertex = self.sequence[rho - 1]
            j: vx.Vertex = self.sequence[rho]

            # trivial feasibility check
            if i.tw.e < u.tw.l and u.tw.e < j.tw.l:

                # compute insertion cost
                dist_i_u = self.distance_matrix.loc[i.id_, u.id_]
                dist_u_j = self.distance_matrix.loc[u.id_, j.id_]
                insertion_cost = dist_i_u + dist_u_j

                if verbose > 2:
                    print(f'Between {i.id_} and {j.id_}: {insertion_cost}')

                if insertion_cost < min_insertion_cost:

                    # check feasibility
                    temp_tour = deepcopy(self)
                    temp_tour.insert_and_reset(index=rho, vertex=u)
                    if temp_tour.is_feasible():
                        # update best known insertion position
                        min_insertion_cost = insertion_cost
                        i_best = i
                        j_best = j
                        insertion_position = rho

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
        c1 criterion of Solomon's I1 insertion heuristic Does NOT include a feasibility check Following the
        description by (Bräysy, Olli; Gendreau, Michel (2005): Vehicle Routing Problem with Time Windows,
        Part I: Route Construction and Local Search Algorithms. In Transportation Science 39 (1), pp. 104–118. DOI:
        10.1287/trsc.1030.0056.)
        """
        alpha_2 = 1 - alpha_1

        i = self.sequence[i_index]
        j = self.sequence[j_index]

        # compute c11
        c11 = self._c11(i, u, j, mu)

        # compute c12
        c12 = self._c12(j_index, u)

        # compute and return c1
        return alpha_1 * c11 + alpha_2 * c12

    def _c11(self, i, u, j, mu):
        c11 = self.distance_matrix.loc[i.id_, u.id_] + self.distance_matrix.loc[u.id_, j.id_] - mu * \
              self.distance_matrix.loc[i.id_, j.id_]
        return c11

    def _c12(self, j_index, u):
        service_start_j = self.service_schedule[j_index]
        temp_tour = deepcopy(self)
        temp_tour.insert_and_reset(index=j_index, vertex=u)
        temp_tour.compute_cost_and_schedules(ignore_tw=True)
        service_start_j_new = temp_tour.service_schedule[j_index + 1]
        c12 = service_start_j_new - service_start_j
        return c12

    def c2(self,
           u: vx.Vertex,
           c1: float,

           lambda_: float = opts['lambda'],
           ):
        """
        c2 criterion of Solomon's I1 insertion heuristic Does NOT include a feasibility check Following the
        description by (Bräysy, Olli; Gendreau, Michel (2005): Vehicle Routing Problem with Time Windows,
        Part I: Route Construction and Local Search Algorithms. In Transportation Science 39 (1), pp. 104–118. DOI:
        10.1287/trsc.1030.0056.)
        """
        depot_id = self.sequence[0].id_
        return lambda_ * self.distance_matrix.loc[depot_id, u.id_] - c1

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
            i = self.sequence[rho - 1]
            j = self.sequence[rho]

            # trivial feasibility check
            if i.tw.e < u.tw.l and u.tw.e < j.tw.l:

                # proper feasibility check
                # TODO: check Solomon (1987) for an efficient and sufficient feasibility check
                temp_tour = deepcopy(self)
                temp_tour.insert_and_reset(index=rho, vertex=u)
                if temp_tour.is_feasible():
                    # compute c1 and c2 and update their best values
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

    def finalize(self, visitor):
        """apply visitor's local search procedure to improve the result after the routing itself has been done"""
        assert (not self._finalized), f'Instance has been finalized with {self._finalizing_visitor} already'
        self._finalizing_visitor = visitor
        visitor.finalize_tour(self)
        pass

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
        x = [r.coords.x for r in self.sequence[1:-1]]
        y = [r.coords.y for r in self.sequence[1:-1]]
        requests, = plt.plot(x, y, marker='o', c=color, alpha=alpha, label=self.id_, linestyle='')
        artists.append(requests)

        # plot arrows and annotations
        for i in range(1, len(self)):
            start = self.sequence[i - 1]
            end = self.sequence[i]
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
