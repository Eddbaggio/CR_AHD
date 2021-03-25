import abc
from typing import List
from src.cr_ahd.auction_module.request_selection import find_cheapest_feasible_insertion
from src.cr_ahd.utility_module import utils as ut

TW_LENGTH = 2
ALL_TW = [ut.TimeWindow(e, e + TW_LENGTH) for e in range(ut.opts['start_time'])]


class TWOfferingBehavior(abc.ABC):
    def execute(self, carrier, request, num_tw):
        offered_time_windows = {tw: self._evaluate_time_window(tw, carrier, request) for tw in ALL_TW}
        offered_time_windows = {k: v for k, v in sorted(offered_time_windows.items(), key=lambda item: item[1])}
        return tuple(offered_time_windows.keys())[:num_tw]

    @abc.abstractmethod
    def _evaluate_time_window(self, tw, carrier, request):
        pass


class CheapestTW(TWOfferingBehavior):
    """
    Select time windows based on their cheapest insertion cost for the request.
    """

    def _evaluate_time_window(self, tw, carrier, request):
        pass
