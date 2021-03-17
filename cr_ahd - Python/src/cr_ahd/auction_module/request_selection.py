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


class FiftyPercentHighestMarginalCost(RequestSelectionBehavior):
    """given the current set of routes, returns the n unrouted requests that has the HIGHEST marginal cost. NOTE:
    since routes may not be final, the current highest-marginal-cost request might not have been chosen at an earlier
    or later stage!"""

    def _select_requests(self, carrier) -> List:
        if not carrier.unrouted_requests:
            return []
        selected_requests = []
        for unrouted in carrier.unrouted_requests:
            mc = cost_of_cheapest_insertion(unrouted, carrier)
            selected_requests.append((unrouted, mc))
        selected_requests, marginal_costs = zip(*sorted(selected_requests, key=lambda x: x[1], reverse=True))
        return selected_requests[:int(np.ceil(len(carrier.unrouted_requests) * 0.5))]


class Cluster(RequestSelectionBehavior):
    """
    Based on: Gansterer,M., & Hartl,R.F. (2016). Request evaluation strategies for carriers in auction-based
    collaborations.
    """
    pass


def cost_of_cheapest_insertion(request, carrier):
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
        t.pop_and_update(1)
    return lowest
