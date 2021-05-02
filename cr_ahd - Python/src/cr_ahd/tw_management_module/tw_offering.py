import abc
from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.utility_module import utils as ut
import src.cr_ahd.core_module.tour as tr


class TWOfferingBehavior(abc.ABC):
    def execute(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, request: int):
        pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(request)
        # make sure that the request has not been given a tw yet
        assert (solution.tw_open[delivery_vertex], solution.tw_close[delivery_vertex]) == (ut.START_TIME, ut.END_TIME)

        tw_valuations = [self._evaluate_time_window(instance, solution, carrier, request, tw) for tw in ut.ALL_TW]
        offered_time_windows = list(sorted(zip(tw_valuations, ut.ALL_TW)))
        offered_time_windows = [tw for valuation, tw in offered_time_windows if valuation >= 0]
        return offered_time_windows  # [:num_tw]

    @abc.abstractmethod
    def _evaluate_time_window(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, request: int,
                              tw: ut.TimeWindow):
        pass

    def _find_first_insertion_index(self, solution: slt.GlobalSolution, tw: ut.TimeWindow, tour: tr.Tour):
        """
        based on the current tour, what is the first index in which insertion is feasible based on tw constraints?
        """
        for first_index, vertex in enumerate(tour.routing_sequence[1:], 1):
            if tw.open <= solution.tw_close[vertex]:
                return first_index
        return first_index

    def _find_last_insertion_index(self, solution: slt.GlobalSolution, tw: ut.TimeWindow, tour: tr.Tour,
                                   first_index: int):
        """
        based on the current routing sequence, what is the last index in which insertion is feasible based on tw
        constraints
        """
        for last_index, vertex in enumerate(tour.routing_sequence[first_index:], first_index):
            if tw.close <= solution.tw_open[vertex]:
                return last_index
        return last_index


class FeasibleTW(TWOfferingBehavior):
    def _evaluate_time_window(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, request: int,
                              tw: ut.TimeWindow):
        carrier_ = solution.carriers[carrier]
        pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(request)

        # can the carrier open a new tour and insert the request there?
        if carrier_.num_tours() < instance.carriers_max_num_tours:
            dist = instance.distance([carrier, pickup_vertex], [pickup_vertex, delivery_vertex])
            if ut.travel_time(dist) <= solution.tw_close[delivery_vertex]:
                return 1

        # if no new tour can be built, can the request be inserted into one of the existing ones?
        for tour in carrier_.tours:
            first_index = self._find_first_insertion_index(solution, tw, tour)
            last_index = self._find_last_insertion_index(solution, tw, tour, first_index)
            for pickup_insertion_index in range(first_index):
                for delivery_insertion_index in range(first_index, last_index + 1):
                    if tour.insertion_feasibility_check(instance, solution,
                                                        [pickup_insertion_index, delivery_insertion_index],
                                                        [pickup_vertex, delivery_vertex]):
                        return 1

        # if request cannot be inserted anywhere, return negative valuation
        return -1


