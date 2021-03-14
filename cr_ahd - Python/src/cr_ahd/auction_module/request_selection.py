from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.cr_ahd.solving_module.tour_construction import find_cheapest_feasible_insertion
from src.cr_ahd.utility_module.utils import InsertionError


class RequestSelectionBehavior(ABC):
    def execute(self, instance):
        """
        select a set of requests based on the concrete selection behavior. will retract the requests from the carrier
        and return a dict of lists.

        :param instance:
        :return: dict {carrier_A: List[requests], carrier_B: List[requests]}
        """
        submitted_requests = dict()
        for carrier in instance.carriers:
            submitted_requests[carrier] = self._select_requests(carrier)
            carrier.retract_requests_and_update_routes(submitted_requests[carrier])
            # # TODO: why are these requests not popped inside the _select_requests method?
            # for request in carrier.routed_requests:
            #     routed_tour, routed_index = request.routed_requests
            #     routed_tour.pop_and_update(routed_index)
        # solution.request_selection = self.__class__.__name__
        return submitted_requests

    @abstractmethod
    def _select_requests(self, carrier):
        pass


class SingleLowestMarginalCost(RequestSelectionBehavior):
    """given the current set of routes, returns the single unrouted request that has the lowest marginal cost. NOTE:
    since routes may not be final, the current lowest-marginal-profit request might not have been chosen at an earlier
    or later stage!"""

    def _select_requests(self, carrier) -> List:
        lowest_marginal_cost = np.infty
        selected_request = []
        for unrouted in carrier.unrouted_requests:
            mc = marginal_cost_request(unrouted, carrier)
            if mc < lowest_marginal_cost:
                selected_request = [unrouted]
                lowest_marginal_cost = mc
        return selected_request  # , [lowest_marginal_cost]


def marginal_cost_request(request, carrier):  # rename!
    lowest = float('inf')
    for vehicle in carrier.active_vehicles:
        try:
            _, tour_min = find_cheapest_feasible_insertion(vehicle.tour, request)
            lowest = min(lowest, tour_min)
        except InsertionError:
            continue
    if lowest >= float('inf'):
        t = carrier.inactive_vehicles[0].tour
        t.insert_and_update(1, request)
        lowest = t.cost
    return lowest
