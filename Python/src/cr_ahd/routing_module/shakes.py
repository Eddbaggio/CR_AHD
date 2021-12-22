import random
from abc import ABC, abstractmethod
from typing import final, List

from core_module import instance as it, solution as slt, tour as tr
import logging

logger = logging.getLogger(__name__)


# =====================================================================================================================
# SHAKES
# =====================================================================================================================
class Shake(ABC):
    """

    """

    @abstractmethod
    def execute(self, instance: it.MDPDPTWInstance, carrier: slt.AHDSolution, num_requests: int):
        """
        Perform the Shake step on each tour of the given carrier. Currently, no intra-tour shakes do exist!

        :param instance:
        :param carrier:
        :param num_requests:
        :return:
        """
        pass

    @abstractmethod
    def execute_on_tour(self, instance: it.MDPDPTWInstance, tour: tr.Tour, num_requests: int):
        """
        Perform the shaking operation on the given tour

        :param instance:
        :param tour:
        :param num_requests:
        :return:
        """
        pass


class RandomRemovalShake(Shake):
    """
    removes num_requests random requests from each given tour
    """

    def execute(self, instance: it.MDPDPTWInstance, carrier: slt.AHDSolution, num_requests: int):
        for tour in carrier.tours:
            removed = self.execute_on_tour(instance, tour, num_requests)
            for request in removed:
                carrier.unrouted_requests.append(request)
                carrier.routed_requests.remove(request)

    def execute_on_tour(self, instance: it.MDPDPTWInstance, tour: tr.Tour, num_requests: int):
        num_requests = min(num_requests, round(len(tour)/2)-1)
        # TODO randomize the num_request by replacing the num_request parameter by max_num_requests and randomly choose
        #  from [1, max_num_requests]
        if num_requests > 0:
            routed = []
            for vertex in tour.routing_sequence[1:-1]:
                request = instance.request_from_vertex(vertex)
                if request not in routed:
                    routed.append(request)

            removed = random.sample(routed, num_requests)
            removal_indices = []
            for request in removed:
                pickup, delivery = instance.pickup_delivery_pair(request)
                removal_indices.append(tour.vertex_pos[pickup])
                removal_indices.append(tour.vertex_pos[delivery])
                tour.requests.remove(request)
            tour.pop_and_update(instance, sorted(removal_indices))
            return removed
        else:
            return []


# =====================================================================================================================
# LNS REMOVAL - OLD
# =====================================================================================================================
'''
class LNSRemoval(ABC):
    @final
    def _execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carriers: List[int] = None,
                 num_removal_requests: int = 1, p: float = float('inf')):
        """
        DO NOT CALL!
        removes num_removal_requests requests from each carrier. some randomness that is introduced by the parameter p.
        removal happens in place

        :param instance:
        :param solution:
        :param num_removal_requests:
        :param p: introduces randomness to the selection of requests. a low value of p corresponds to much randomness
        :return: the removed requests as a list
        """
        if carriers is None:
            carriers = range(len(solution.carriers))

        all_removal_requests = []

        for carrier in carriers:
            carrier_ = solution.carriers[carrier]
            removal_requests = self.select_removal_requests_from_carrier(instance, solution, carrier_,
                                                                         num_removal_requests, p)
            # do the actual removal/un-routing todo: this should be the execute_move part of the neighborhood
            for request in removal_requests:
                pickup, delivery = instance.pickup_delivery_pair(request)
                tour_ = carrier_.tours[solution.request_to_tour_assignment[request]]
                pickup_pos = solution.vertex_position_in_tour[pickup]
                delivery_pos = solution.vertex_position_in_tour[delivery]
                tour_.pop_and_update(instance, solution, [pickup_pos, delivery_pos])

            all_removal_requests.extend(removal_requests)

        return all_removal_requests

    @abstractmethod
    def select_removal_requests_from_carrier(self, instance: it.PDPInstance, solution: slt.CAHDSolution,
                                             carrier_: slt.AHDSolution,
                                             num_removal_requests: int, p: float = float('inf')):
        pass


class ShawRemoval(LNSRemoval):
    """
    request selection happens based on similarity
    """

    def select_removal_requests_from_carrier(self, instance: it.PDPInstance, solution: slt.CAHDSolution,
                                             carrier_: slt.AHDSolution,
                                             num_removal_requests: int, p: float = float('inf')):
        assert p >= 1
        assert num_removal_requests <= instance.num_requests

        fixed_requests = carrier_.accepted_requests.copy()
        # select random initial request
        r = random.choice(carrier_.accepted_requests)
        removal_requests = [r]
        fixed_requests.remove(r)
        # select similar requests
        while len(removal_requests) < num_removal_requests:
            r = random.choice(removal_requests)
            fixed_requests.sort(
                key=lambda x: bv.ropke_pisinger_request_similarity(instance, solution, r, x, capacity_weight=0))
            i = round(random.random() ** p * len(fixed_requests))  # randomized index of the selected request
            removal_requests.append(fixed_requests[i])
            fixed_requests.pop(i)
        return removal_requests


class RandomRemoval(LNSRemoval):
    """
    removal requests are selected randomly from the accepted requests
    """

    def select_removal_requests_from_carrier(self, instance: it.PDPInstance, solution: slt.CAHDSolution,
                                             carrier_: slt.AHDSolution,
                                             num_removal_requests: int, p: float = float('inf')):
        assert p >= 1
        assert num_removal_requests <= instance.num_requests

        removal_requests = random.sample(carrier_.accepted_requests, num_removal_requests)
        return removal_requests


class WorstRemoval(LNSRemoval):
    def select_removal_requests_from_carrier(self, instance: it.PDPInstance, solution: slt.CAHDSolution,
                                             carrier_: slt.AHDSolution,
                                             num_removal_requests: int, p: int):
        raise NotImplementedError
        pass
'''