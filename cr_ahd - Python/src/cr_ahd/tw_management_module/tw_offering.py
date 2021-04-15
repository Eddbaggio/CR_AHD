import abc

from src.cr_ahd.utility_module import utils as ut
import src.cr_ahd.core_module.tour as tr


class TWOfferingBehavior(abc.ABC):
    def execute(self, carrier, request):
        assert request.tw == ut.TIME_HORIZON
        offered_time_windows = {tw: self._evaluate_time_window(tw, carrier, request) for tw in ut.ALL_TW}
        offered_time_windows = {k: v for k, v in offered_time_windows.items() if bool(v)}  # only positive evaluations
        offered_time_windows = {k: v for k, v in sorted(offered_time_windows.items(), key=lambda item: item[1])}
        return tuple(offered_time_windows.keys())  # [:num_tw]

    @abc.abstractmethod
    def _evaluate_time_window(self, tw, carrier, request):
        pass

    def _find_first_insertion_index(self, tw: ut.TimeWindow, tour: tr.Tour):
        for first_index, request in enumerate(tour.routing_sequence[1:], 1):
            if tw.e <= request.tw.l:
                return first_index
        return first_index

    def _find_last_insertion_index(self, tw, tour, first_index):
        for last_index, request in enumerate(tour.routing_sequence[first_index:], first_index):
            if tw.l <= request.tw.e:
                return last_index
        return last_index


class FeasibleTW(TWOfferingBehavior):

    def _evaluate_time_window(self, tw: ut.TimeWindow, carrier, request):
        feasible_flag = False
        for vehicle in [*carrier.active_vehicles, carrier.inactive_vehicles[0]]:  # TODO: what about initializing a new route?!
            first_index = self._find_first_insertion_index(tw, vehicle.tour)
            last_index = self._find_last_insertion_index(tw, vehicle.tour, first_index)
            for insertion_index in range(first_index, last_index + 1):
                try:
                    vehicle.tour.insert_and_update(insertion_index, request)
                    vehicle.tour.pop_and_update(insertion_index)
                    feasible_flag = True
                    return feasible_flag
                except ut.InsertionError:
                    continue
        return feasible_flag


class CheapestTW(TWOfferingBehavior):
    """
    Select time windows based on their cheapest insertion cost for the request.
    """

    def _evaluate_time_window(self, tw, carrier, request):
        pass
