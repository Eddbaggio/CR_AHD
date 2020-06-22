import vertex as vx
from utils import travel_time, opts, InsertionError
from copy import deepcopy
from typing import List


class Tour(object):
    def __init__(self, id_: str, sequence: List[vx.Vertex]):
        self.id_ = id_
        self.sequence = sequence    # sequence of vertices
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

    def copy(self):
        tour = Tour(self.sequence)
        tour.arrival_schedule = self.arrival_schedule
        tour.service_schedule = self.service_schedule
        tour.cost = self.cost
        return tour

    def insert_and_reset_schedules(self, index: int, vertex: vx.Vertex):
        """ inserts a vertex before the specified index and resets all schedules to None. """
        self.sequence.insert(index, vertex)
        self.arrival_schedule = [None] * len(self)  # reset arrival times
        self.service_schedule = [None] * len(self)  # reset start of service times
        pass

    # TODO: custom versions of the list methods, e.g. insert method that automatically updates the schedules?!
    def _insert_and_update_schedules(self):
        pass

    def compute_cost_and_schedules(self, dist_matrix, start_time=opts['start_time'], ignore_tw=True, verbose=opts['verbose']):
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
        if verbose > 1:
            print(f'== Feasibility Check')
            print(self)
        service_schedule = self.service_schedule.copy()
        service_schedule[0] = start_time
        for rho in range(1, len(self)):
            i: vx.Vertex = self.sequence[rho - 1]
            j: vx.Vertex = self.sequence[rho]
            dist = dist_matrix.loc[i.id_, j.id_]
            if verbose > 2:
                print(f'iteration {rho}; service_schedule: {service_schedule}')
            earliest_arrival = service_schedule[rho - 1] + i.service_duration + travel_time(dist)
            if earliest_arrival > j.tw.l:
                if verbose > 0:
                    print(f'From {i} to {j}')
                    print(f'Predecessor start of service : {service_schedule[rho - 1]}')
                    print(f'Predecessor service time : {i.service_duration}')
                    print(f'Distance : {dist}')
                    print(f'Travel time: {travel_time(dist)}')
                    print(f'Infeasible! {round(earliest_arrival, 2)} > {j.id_}.tw.l: {j.tw.l}')
                    print()
                return False
            else:
                service_schedule[rho] = max(j.tw.e, earliest_arrival)
        return True

    def cheapest_feasible_insertion(self, u: vx.Vertex, dist_matrix, verbose=opts['verbose']):
        """calculating the cheapest insertion postition for an unrouted customer,
        returning cost and position"""
        if verbose > 0:
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

                if verbose > 1:
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
            else:
                print('skip')

        # return the best found position and cost or raise an error if no feasible position was found
        if i_best:
            if verbose > 0:
                print(f'== Best: between {i_best.id_} and {j_best.id_}: {min_insertion_cost}')
            return insertion_position, min_insertion_cost
        else:
            raise InsertionError('', 'No feasible insertion position found')
        pass
