import abc
import datetime as dt
from typing import Sequence, List
from src.cr_ahd.core_module import instance as it, solution as slt, tour as tr
from src.cr_ahd.utility_module import utils as ut


class TWOfferingBehavior(abc.ABC):
    def execute(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution, carrier: int, request: int):
        pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(request)
        # make sure that the request has not been given a tw yet
        assert (instance.tw_open[delivery_vertex], instance.tw_close[delivery_vertex]) in [(ut.START_TIME, ut.END_TIME),
                                                                                           (None, None)]

        tw_valuations = [self._evaluate_time_window(instance, solution, carrier, request, tw) for tw in ut.ALL_TW]
        offered_time_windows = list(
            sorted(zip(tw_valuations, ut.ALL_TW), key=lambda x: x[0]))
        offered_time_windows = [tw for valuation, tw in offered_time_windows if valuation >= 0]
        return offered_time_windows  # [:num_tw]

    @abc.abstractmethod
    def _evaluate_time_window(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution, carrier: int,
                              request: int,
                              tw: ut.TimeWindow):
        pass


"""
class FeasibleTW(TWOfferingBehavior):
    def _evaluate_time_window(self, instance: it.PDPInstance,
                              solution: slt.CAHDSolution,
                              carrier: int,
                              request: int,
                              tw: ut.TimeWindow):
        carrier_ = solution.carriers[carrier]
        pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(request)
        instance.tw_open[delivery_vertex] = tw.open
        instance.tw_close[delivery_vertex] = tw.close

        # can the carrier open a new tour and insert the request there? only checks the time window constraint.
        # it is assumed that load and max tour length are not exceeded with a single request
        if carrier_.num_tours() < instance.carriers_max_num_tours:
            new_tour_feasible = True
            service_time = ut.START_TIME
            for predecessor_vertex, vertex in zip([carrier, pickup_vertex, delivery_vertex],
                                                  [pickup_vertex, delivery_vertex, carrier]):

                dist = instance.distance([predecessor_vertex], [vertex])
                arrival_time = service_time + instance.service_duration[predecessor_vertex] + ut.travel_time(dist)

                if arrival_time > instance.tw_close[vertex]:
                    new_tour_feasible = False
                    break

                service_time = max(arrival_time, instance.tw_open[vertex])

            if new_tour_feasible:
                # undo the setting of the time window and return
                instance.tw_open[delivery_vertex] = ut.START_TIME
                instance.tw_close[delivery_vertex] = ut.END_TIME
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
                        instance.tw_open[delivery_vertex] = ut.START_TIME
                        instance.tw_close[delivery_vertex] = ut.END_TIME
                        return 1

        # if request cannot be inserted anywhere return negative valuation
        # undo the setting of the time window and return
        instance.tw_open[delivery_vertex] = ut.START_TIME
        instance.tw_close[delivery_vertex] = ut.END_TIME
        return -1
"""


class FeasibleTW(TWOfferingBehavior):
    def _evaluate_time_window(
            self,
            instance: it.MDPDPTWInstance,
            solution: slt.CAHDSolution,
            carrier: int,
            request: int,
            tw: ut.TimeWindow
    ):
        """
        :return: 1 if TW is feasible, -1 else
        """

        carrier_ = solution.carriers[carrier]
        pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(request)

        # can the carrier open a new pendulum tour and insert the request there? only checks the time window constraint.
        # it is assumed that load and max tour length are not exceeded with a single request
        if carrier_.num_tours() < instance.carriers_max_num_tours:

            new_tour_feasible = True
            service_time = ut.START_TIME

            for predecessor_vertex, vertex in zip([carrier_.id_, pickup_vertex, delivery_vertex],
                                                  [pickup_vertex, delivery_vertex, carrier_.id_]):

                arrival_time = service_time + \
                               instance.vertex_service_duration[predecessor_vertex] + \
                               instance.travel_duration([predecessor_vertex], [vertex])

                if arrival_time > instance.tw_close[vertex]:
                    new_tour_feasible = False
                    break

                service_time = max(arrival_time, instance.tw_open[vertex])

            if new_tour_feasible:
                return 1

        # if no new tour can be built, can the request be inserted into one of the existing tours?
        else:
            # temporarily set the time window
            instance.tw_open[delivery_vertex] = tw.open
            instance.tw_close[delivery_vertex] = tw.close

            for tour_ in carrier_.tours(solution):
                for pickup_pos in range(1, len(tour_)):
                    for delivery_pos in range(pickup_pos + 1, len(tour_) + 1):
                        if tour_.insertion_feasibility_check(instance, [pickup_pos, delivery_pos],
                                                             [pickup_vertex, delivery_vertex]):
                            # undo the setting of the time window and return
                            instance.tw_open[delivery_vertex] = ut.START_TIME
                            instance.tw_close[delivery_vertex] = ut.END_TIME
                            return 1

            # if request cannot be inserted anywhere return negative valuation
            # undo the setting of the time window and return
            instance.tw_open[delivery_vertex] = ut.START_TIME
            instance.tw_close[delivery_vertex] = ut.END_TIME
            return -1
