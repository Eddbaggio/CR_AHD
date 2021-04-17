import itertools
import logging
from abc import ABC, abstractmethod

import numpy as np

from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import tour_construction as cns

logger = logging.getLogger(__name__)


class RequestSelectionBehavior(ABC):
    def execute(self, instance: it.PDPInstance, solution: slt.GlobalSolution, num_requests):
        """
        select a set of requests based on the concrete selection behavior. will retract the requests from the carrier
        and return a dict of lists.

        :param solution:
        :param num_requests: the number of requests each carrier shall submit. If fractional, will use the corresponding
         percentage of requests assigned to that carrier. If integer, will use the absolut amount of requests. If not
         enough requests are available, will submit as many as are available
        :param instance:
        :return: dict {carrier_A: List[requests], carrier_B: List[requests]}
        """
        submitted_requests = list()
        for carrier in range(instance.num_carriers):
            k = self.abs_num_requests(solution.carrier_solutions[carrier], num_requests)
            valuations = self._evaluate_requests(instance, solution, carrier)
            # pick the worst k evaluated requests (from ascending order)
            selected = [r for _, r in
                        sorted(zip(valuations, solution.carrier_solutions[carrier].unrouted_requests))][:k]
            for s in selected:
                solution.carrier_solutions[carrier].unrouted_requests.remove(s)  # TODO it's not routed yet! why do I already remove it here?
                solution.request_to_carrier_assignment[s] = np.nan
                submitted_requests.append(s)
                solution.unassigned_requests.append(s)  # if i do this, the assign_n_requests function will not work
        return submitted_requests

    @abstractmethod
    def _evaluate_requests(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int):
        """
        compute the carrier's valuation for all unrouted requests of that carrier.
        NOTE: a high number corresponds to a high valuation

        """
        pass

    def abs_num_requests(self, carrier: slt.PDPSolution, num_requests) -> int:
        """
        returns the absolute number of requests that a carrier shall submit, depending on whether it was initially
        given as an absolute int or a float (relative)
        """
        if isinstance(num_requests, int):
            return num_requests
        elif isinstance(num_requests, float):
            assert num_requests <= 1, 'If providing a float, must be <=1 to be converted to percentage'
            return round(len(carrier.unrouted_requests) * num_requests)


class Random(RequestSelectionBehavior):
    """
    returns a random selection of unrouted requests
    """

    def _evaluate_requests(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int):
        return np.random.random(len(solution.carrier_solutions[carrier].unrouted_requests))


class HighestInsertionCostDistance(RequestSelectionBehavior):
    """given the current set of routes, returns the n unrouted requests that have the HIGHEST Insertion distance cost.
     NOTE: since routes may not be final, the current highest-marginal-cost request might not have been chosen at an
     earlier or later stage!"""

    def _evaluate_requests(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int):
        evaluation = []
        cs = solution.carrier_solutions[carrier]
        for request in cs.unrouted_requests:
            best_delta_for_r = float('inf')
            for tour in range(cs.num_tours()):
                # cheapest way to fit request into tour
                delta, _, _ = cns.CheapestInsertion()._tour_cheapest_insertion(instance, solution, carrier, tour,
                                                                               request)
                if delta < best_delta_for_r:
                    best_delta_for_r = delta
            # if no feasible insertion for the current request was found, attempt to create a new tour, if that's not
            # feasible the best_delta_for_r will be infinity
            if best_delta_for_r == float('inf'):
                if cs.num_tours() < instance.carriers_max_num_tours:
                    pickup, delivery = instance.pickup_delivery_pair(request)
                    best_delta_for_r = instance.distance([carrier, pickup, delivery], [pickup, delivery, carrier])
            # collect the NEGATIVE value, since high insertion cost mean a low valuation for the carrier
            evaluation.append(-best_delta_for_r)
        return evaluation


class LowestProfit(RequestSelectionBehavior):
    def _evaluate_requests(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int):
        revenue = []
        for r in solution.carrier_solutions[carrier].unrouted_requests:
            request_revenue = sum([instance.revenue[v] for v in instance.pickup_delivery_pair(r)])
            revenue.append(request_revenue)
        ins_cost = HighestInsertionCostDistance()._evaluate_requests(instance, solution, carrier)
        # return the profit, high profit means high valuation
        return [rev + cost for rev, cost in zip(revenue, ins_cost)]


'''
class Cluster(RequestSelectionBehavior):
    """
    Based on: Gansterer,M., & Hartl,R.F. (2016). Request evaluation strategies for carriers in auction-based
    collaborations.
    Uses geographical information (intra-cluster distances of cluster members) to select requests that are in close
    proximity
    """

    def _evaluate_requests(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int):
        cs = solution.carrier_solutions[carrier]
        candidate_clusters = itertools.combinations(cs.unrouted_requests, solution.num_carriers())
        best_cluster, best_cluster_evaluation = [], float('inf')
        for candidate in candidate_clusters:
            evaluation = self.cluster_evaluation(instance, solution, carrier, candidate)
            if evaluation < best_cluster_evaluation:
                best_cluster, best_cluster_evaluation = candidate, evaluation
        if best_cluster_evaluation < float('inf'):
            return best_cluster  # , best_cluster_evaluation
        else:
            raise ValueError()

    def cluster_evaluation(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, cluster):
        """
        sum of the distances of (pickup_0, pickup_1) and (delivery_0, delivery_1)
        """
        pairs = itertools.combinations(cluster, 2)
        evaluation = 0
        for r0, r1 in pairs:
            evaluation += instance.distance(instance.pickup_delivery_pair(r0), instance.pickup_delivery_pair(r1))
        return evaluation
        '''
