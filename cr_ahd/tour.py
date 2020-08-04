from copy import deepcopy
from typing import List

import matplotlib.pyplot as plt

import vertex as vx
from utils import travel_time, opts, InsertionError


class Tour(object):
    def __init__(self, id_: str, sequence: List[vx.Vertex]):
        self.id_ = id_
        self.sequence = sequence  # sequence of vertices
        self.depot = sequence[0]
        # TODO: should this really be a sequence? What about the successor/adjacency-list?
        self.arrival_schedule = [None] * len(sequence)  # arrival times
        self.service_schedule = [None] * len(sequence)  # start of service times
        self.cost = 0
        pass

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

    # def copy(self):
    #     tour = Tour(self.sequence)
    #     tour.arrival_schedule = self.arrival_schedule
    #     tour.service_schedule = self.service_schedule
    #     tour.cost = self.cost
    #     return tour

    def insert_and_reset_schedules(self, index: int, vertex: vx.Vertex):
        """ inserts a vertex before the specified index and resets all schedules to None. """
        assert type(vertex) == vx.Vertex
        self.sequence.insert(index, vertex)
        self.arrival_schedule = [None] * len(self)  # reset arrival times
        self.service_schedule = [None] * len(self)  # reset start of service times
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

    def compute_cost_and_schedules(self, dist_matrix, start_time=opts['start_time'], ignore_tw=True,
                                   verbose=opts['verbose']):
        self.cost = 0
        self.arrival_schedule[0] = start_time
        self.service_schedule[0] = start_time
        for rho in range(1, len(self)):
            i: vx.Vertex = self.sequence[rho - 1]
            j: vx.Vertex = self.sequence[rho]
            dist = dist_matrix.loc[i.id_, j.id_]
            self.cost += dist
            planned_arrival = self.service_schedule[rho - 1] + i.service_duration + travel_time(dist)
            if verbose > 2:
                print(f'Planned arrival at {j}: {planned_arrival}')
            if not ignore_tw:
                assert planned_arrival <= j.tw.l
            self.arrival_schedule[rho] = planned_arrival
            if planned_arrival >= j.tw.e:
                self.service_schedule[rho] = planned_arrival
            else:
                self.service_schedule[rho] = j.tw.e
        pass

    def is_feasible(self, dist_matrix, start_time=opts['start_time'], verbose=opts['verbose']) -> bool:
        if verbose > 2:
            print(f'== Feasibility Check')
            print(self)
        service_schedule = self.service_schedule.copy()
        service_schedule[0] = start_time
        for rho in range(1, len(self)):
            i: vx.Vertex = self.sequence[rho - 1]
            j: vx.Vertex = self.sequence[rho]
            dist = dist_matrix.loc[i.id_, j.id_]
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

    def find_cheapest_feasible_insertion(self, u: vx.Vertex, dist_matrix, verbose=opts['verbose']):
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
                dist_i_u = dist_matrix.loc[i.id_, u.id_]
                dist_u_j = dist_matrix.loc[u.id_, j.id_]
                insertion_cost = dist_i_u + dist_u_j

                if verbose > 2:
                    print(f'Between {i.id_} and {j.id_}: {insertion_cost}')

                if insertion_cost < min_insertion_cost:

                    # check feasibility
                    temp_tour = deepcopy(self)
                    temp_tour.insert_and_reset_schedules(index=rho, vertex=u)
                    if temp_tour.is_feasible(dist_matrix=dist_matrix):
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
           dist_matrix, ) -> float:
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
        c11 = self._c11(dist_matrix, i, u, j, mu)

        # compute c12
        c12 = self._c12(dist_matrix, j_index, u)

        # compute and return c1
        return alpha_1 * c11 + alpha_2 * c12

    @staticmethod
    def _c11(dist_matrix, i, u, j, mu):
        c11 = dist_matrix.loc[i.id_, u.id_] + dist_matrix.loc[u.id_, j.id_] - mu * dist_matrix.loc[i.id_, j.id_]
        return c11

    def _c12(self, dist_matrix, j_index, u):
        service_start_j = self.service_schedule[j_index]
        temp_tour = deepcopy(self)
        temp_tour.insert_and_reset_schedules(index=j_index, vertex=u)
        temp_tour.compute_cost_and_schedules(dist_matrix=dist_matrix, ignore_tw=True)
        service_start_j_new = temp_tour.service_schedule[j_index + 1]
        c12 = service_start_j_new - service_start_j
        return c12

    def c2(self,
           u: vx.Vertex,
           c1: float,
           dist_matrix,
           lambda_: float = opts['lambda'],
           ):
        """
        c2 criterion of Solomon's I1 insertion heuristic Does NOT include a feasibility check Following the
        description by (Bräysy, Olli; Gendreau, Michel (2005): Vehicle Routing Problem with Time Windows,
        Part I: Route Construction and Local Search Algorithms. In Transportation Science 39 (1), pp. 104–118. DOI:
        10.1287/trsc.1030.0056.)
        """
        depot_id = self.sequence[0].id_
        return lambda_ * dist_matrix.loc[depot_id, u.id_] - c1

    def find_best_feasible_I1_insertion(self, u: vx.Vertex, dist_matrix, verbose=opts['verbose']):
        """
        returns float('-inf') if no feasible insertion position was found
        :param u:
        :param dist_matrix:
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
                temp_tour.insert_and_reset_schedules(index=rho, vertex=u)
                if temp_tour.is_feasible(dist_matrix=dist_matrix):
                    # compute c1 and c2 and update their best values
                    c1 = self.c1(i_index=rho - 1,
                                 u=u,
                                 j_index=rho,
                                 alpha_1=opts['alpha_1'],
                                 mu=opts['mu'],
                                 dist_matrix=dist_matrix)
                    if verbose > 1:
                        print(f'c1({u.id_}->{self.id_}): {c1}')
                    c2 = self.c2(u=u, lambda_=opts['lambda'], c1=c1, dist_matrix=dist_matrix)
                    if verbose > 1:
                        print(f'c2({u.id_}->{self.id_}): {c2}')
                    if c2 > max_c2:
                        max_c2 = c2
                        rho_best = rho
        return rho_best, max_c2

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
                                           xy=(end.coords.x, end.coords.y+1),
                                           alpha=alpha
                                           )
                artists.append(annotations)
        return artists
