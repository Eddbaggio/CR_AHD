import abc
import datetime as dt
from typing import Sequence, List
from src.cr_ahd.core_module import instance as it, solution as slt, tour as tr
from src.cr_ahd.utility_module import utils as ut


class TWOfferingBehavior(abc.ABC):
    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, request: int):
        pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(request)
        # make sure that the request has not been given a tw yet
        assert (solution.tw_open[delivery_vertex], solution.tw_close[delivery_vertex]) == (ut.START_TIME, ut.END_TIME)

        tw_valuations = [self._evaluate_time_window(instance, solution, carrier, request, tw) for tw in ut.ALL_TW]
        offered_time_windows = list(
            sorted(zip(tw_valuations, ut.ALL_TW), key=lambda x: x[0]))
        offered_time_windows = [tw for valuation, tw in offered_time_windows if valuation >= 0]
        return offered_time_windows  # [:num_tw]

    @abc.abstractmethod
    def _evaluate_time_window(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, request: int,
                              tw: ut.TimeWindow):
        pass

    def _find_first_insertion_index(self, solution: slt.CAHDSolution, tw: ut.TimeWindow, tour: tr.Tour):
        """
        based on the current tour, what is the first index in which insertion is feasible based on tw constraints?
        """
        for first_index, vertex in enumerate(tour.routing_sequence[1:], 1):
            if tw.open <= solution.tw_close[vertex]:
                return first_index
        return first_index

    def _find_last_insertion_index(self, solution: slt.CAHDSolution, tw: ut.TimeWindow, tour: tr.Tour,
                                   first_index: int):
        """
        based on the current routing sequence, what is the last index in which insertion is feasible based on tw
        constraints
        """
        for last_index, vertex in enumerate(tour.routing_sequence[first_index:], first_index):
            if tw.close <= solution.tw_open[vertex]:
                return last_index
        return last_index


"""
class FeasibleTW(TWOfferingBehavior):
    def _evaluate_time_window(self, instance: it.PDPInstance,
                              solution: slt.CAHDSolution,
                              carrier: int,
                              request: int,
                              tw: ut.TimeWindow):
        carrier_ = solution.carriers[carrier]
        pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(request)
        solution.tw_open[delivery_vertex] = tw.open
        solution.tw_close[delivery_vertex] = tw.close

        # can the carrier open a new tour and insert the request there? only checks the time window constraint.
        # it is assumed that load and max tour length are not exceeded with a single request
        if carrier_.num_tours() < instance.carriers_max_num_tours:
            new_tour_feasible = True
            service_time = ut.START_TIME
            for predecessor_vertex, vertex in zip([carrier, pickup_vertex, delivery_vertex],
                                                  [pickup_vertex, delivery_vertex, carrier]):

                dist = instance.distance([predecessor_vertex], [vertex])
                arrival_time = service_time + instance.service_duration[predecessor_vertex] + ut.travel_time(dist)

                if arrival_time > solution.tw_close[vertex]:
                    new_tour_feasible = False
                    break

                service_time = max(arrival_time, solution.tw_open[vertex])

            if new_tour_feasible:
                # undo the setting of the time window and return
                solution.tw_open[delivery_vertex] = ut.START_TIME
                solution.tw_close[delivery_vertex] = ut.END_TIME
                return 1

        # if no new tour can be built, can the request be inserted into one of the existing ones?
        for tour_ in carrier_.tours:
            # first_index = self._find_first_insertion_index(solution, tw, tour_)
            # last_index = self._find_last_insertion_index(solution, tw, tour_, first_index)
            for pickup_pos in range(1, len(tour_)):
                for delivery_pos in range(pickup_pos + 1, len(tour_) + 1):
                    if tour_.multi_insertion_feasibility_check(instance, solution,
                                                         [pickup_pos, delivery_pos],
                                                         [pickup_vertex, delivery_vertex]):
                        # undo the setting of the time window and return
                        solution.tw_open[delivery_vertex] = ut.START_TIME
                        solution.tw_close[delivery_vertex] = ut.END_TIME
                        return 1

        # if request cannot be inserted anywhere return negative valuation
        # undo the setting of the time window and return
        solution.tw_open[delivery_vertex] = ut.START_TIME
        solution.tw_close[delivery_vertex] = ut.END_TIME
        return -1
"""


class FeasibleTW(TWOfferingBehavior):
    def _evaluate_time_window(
            self,
            instance: it.PDPInstance,
            solution: slt.CAHDSolution,
            carrier: int,
            request: int,
            tw: ut.TimeWindow
    ):
        """
        This method must work as a collaborative problem but also as a multi-depot problem with only a single carrier!

        :return: 1 if TW is feasible, -1 else
        """
        # instance must have 1 depot per carrier (collab) or only one carrier (multi-depot)
        assert instance.num_depots == instance.num_carriers or instance.num_carriers == 1

        carrier_ = solution.carriers[carrier]
        pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(request)
        solution.tw_open[delivery_vertex] = tw.open
        solution.tw_close[delivery_vertex] = tw.close

        # can the carrier open a new tour and insert the request there? only checks the time window constraint.
        # it is assumed that load and max tour length are not exceeded with a single request
        if carrier_.num_tours() < instance.carriers_max_num_tours * (len(solution.carrier_depots[carrier])):

            for depot in solution.carrier_depots[carrier]:
                new_tour_feasible = True
                service_time = ut.START_TIME

                for predecessor_vertex, vertex in zip([depot, pickup_vertex, delivery_vertex],
                                                      [pickup_vertex, delivery_vertex, depot]):

                    dist = instance.distance([predecessor_vertex], [vertex])
                    arrival_time = service_time + instance.service_duration[predecessor_vertex] + ut.travel_time(dist)

                    if arrival_time > solution.tw_close[vertex]:
                        new_tour_feasible = False
                        break

                    service_time = max(arrival_time, solution.tw_open[vertex])

                if new_tour_feasible:
                    # undo the setting of the time window and return
                    solution.tw_open[delivery_vertex] = ut.START_TIME
                    solution.tw_close[delivery_vertex] = ut.END_TIME
                    return 1

        # if no new tour can be built, can the request be inserted into one of the existing ones?
        for tour_ in carrier_.tours:
            for pickup_pos in range(1, len(tour_)):
                for delivery_pos in range(pickup_pos + 1, len(tour_) + 1):
                    if tour_.insertion_feasibility_check(instance, solution,
                                                         [pickup_pos, delivery_pos],
                                                         [pickup_vertex, delivery_vertex]):
                        # undo the setting of the time window and return
                        solution.tw_open[delivery_vertex] = ut.START_TIME
                        solution.tw_close[delivery_vertex] = ut.END_TIME
                        return 1

        # if request cannot be inserted anywhere return negative valuation
        # undo the setting of the time window and return
        solution.tw_open[delivery_vertex] = ut.START_TIME
        solution.tw_close[delivery_vertex] = ut.END_TIME
        return -1
