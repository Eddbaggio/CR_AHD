import datetime as dt
import random
from abc import ABC, abstractmethod
from math import sqrt
from typing import Tuple

import numpy as np

from auction_module.request_selection.request_selection import RequestSelectionBehavior, _abs_num_requests
from core_module import instance as it, solution as slt, tour as tr
from routing_module import tour_construction as cns, metaheuristics as mh, neighborhoods as ls


class RequestSelectionBehaviorIndividual(RequestSelectionBehavior, ABC):
    """
    select (for each carrier) a set of bundles based on their individual evaluation of some quality criterion
    """

    def execute(self, instance: it.CAHDInstance, solution: slt.CAHDSolution):
        """
        select a set of requests based on the concrete selection behavior. will retract the requests from the carrier
        and return

        :return: the auction_request_pool as a list of request indices and a default
        partition, i.e. a list of the carrier indices that maps the auction_request_pool to their original carrier.
        """
        auction_request_pool = []
        original_partition_labels = []
        for carrier in solution.carriers:
            k = _abs_num_requests(carrier, self.num_submitted_requests)
            valuations = []
            for request in carrier.accepted_requests:
                valuation = self._evaluate_request(instance, solution, carrier, request)
                valuations.append(valuation)

            # pick the k requests with the LOWEST valuation (from ascending order)
            selected = [r for _, r in sorted(zip(valuations, carrier.accepted_requests))][:k]
            selected.sort()

            for request in selected:
                solution.free_requests_from_carriers(instance, [request])
                auction_request_pool.append(request)
                original_partition_labels.append(carrier.id_)

        return auction_request_pool, original_partition_labels

    @abstractmethod
    def _evaluate_request(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, carrier: slt.AHDSolution,
                          request: int):
        """compute the valuation of the given request for the carrier"""
        pass


class Random(RequestSelectionBehaviorIndividual):
    """
    returns a random selection of unrouted requests
    """

    def _evaluate_request(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, carrier: slt.AHDSolution,
                          request: int):
        return random.random()


class MarginalProfit(RequestSelectionBehaviorIndividual):
    """
        "Requests are selected based on their marginal profits. It is calculated by diminishing the revenue of a
    request by the fulfillment cost. The revenue consists of a fixed and a distance-dependent transportation rate for
    traveling from the pickup to the delivery node. The fulfillment costs are composed of fixed stopping costs and
    the marginal travel cost. The latter can easily be determined by calculating the difference in tour lengths for a
    tour that includes, and a tour that excludes this request. Note that for each evaluation the tour-building
    procedure (see Sect. 3.1) has to be called. Thus, this strategy comes with a relatively high computational
    effort. Requests with the lowest marginal profits are submitted to the pool. This strategy is proposed by Berger
    and Bierwirth (2010)."

    NOTE: due to the lack of some functionality (e.g., efficiently identifying all requests that belong to a tour),
    this is not yet functional.
    """

    def _evaluate_request(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, carrier: slt.AHDSolution,
                          request: int):
        raise NotImplementedError

        tour = solution.tour_of_request(request)

        travel_distance_with_request = tour.sum_travel_distance

        # create a new tour without the request
        tmp_tour = tr.Tour('tmp', tour.routing_sequence[0])
        requests_in_tour = tour.requests.copy()
        requests_in_tour.remove(request)

        insertion = cns.MinTravelDistanceInsertion()
        for insert_request in requests_in_tour:
            delta, pickup_pos, delivery_pos = insertion.best_insertion_for_request_in_tour(instance, tmp_tour,
                                                                                           insert_request)
            insertion.execute_insertion_in_tour(instance, solution, tmp_tour, request, pickup_pos, delivery_pos)
        # improvement TODO: tour improvement method should be a parameter
        mh.VRPTWVariableNeighborhoodDescent([ls.PDPMove(), ls.PDPTwoOpt()]).execute_on_tour(instance, tmp_tour)
        travel_distance_without_request = tmp_tour.sum_travel_distance

        return travel_distance_with_request - travel_distance_without_request


class MarginalProfitProxy(RequestSelectionBehaviorIndividual):
    """
    "Requests are selected based on their marginal profits. It is calculated by diminishing the revenue of a
    request by the fulfillment cost. The revenue consists of a fixed and a distance-dependent transportation rate for
    traveling from the pickup to the delivery node. The fulfillment costs are composed of fixed stopping costs and
    the marginal travel cost. The latter can easily be determined by calculating the difference in tour lengths for a
    tour that includes, and a tour that excludes this request. Note that for each evaluation the tour-building
    procedure (see Sect. 3.1) has to be called. Thus, this strategy comes with a relatively high computational
    effort. Requests with the lowest marginal profits are submitted to the pool. This strategy is proposed by Berger
    and Bierwirth (2010)."

    My implementation is not exactly as described, since i do not call the tour_construction twice per request. Instead,
    the tour that excludes the request is simply calculated by the delta in travel distance when removing the request
    from its tour
    """

    def _evaluate_request(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, carrier: slt.AHDSolution,
                          request: int):
        tour = solution.tour_of_request(request)
        delivery = instance.vertex_from_request(request)
        delivery_pos = tour.vertex_pos[delivery]
        # marginal fulfillment cost for the request
        marginal_fulfillment_cost = - tour.pop_distance_delta(instance, [delivery_pos])
        # marginal profit = revenue - fulfillment cost
        marginal_profit = instance.vertex_revenue[delivery] - marginal_fulfillment_cost
        return marginal_profit


class MinDistanceToForeignDepotDMin(RequestSelectionBehaviorIndividual):
    """
    Select the requests that are closest to another carrier's depot. the distance of a request to a depot is the
    MINIMUM of the distances (delivery - depot) and (depot - delivery)
    """

    def _evaluate_request(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, carrier: slt.AHDSolution,
                          request: int):

        foreign_depots = list(range(instance.num_carriers))
        foreign_depots.pop(carrier.id_)

        dist_min = float('inf')
        delivery = instance.vertex_from_request(request)
        for depot in foreign_depots:
            dist = min(instance.travel_distance([depot], [delivery]), instance.travel_distance([delivery], [depot]))
            if dist < dist_min:
                dist_min = dist

        return dist_min


class MinDurationToForeignDepotDMin(RequestSelectionBehaviorIndividual):
    """
    Select the requests that are closest to another carrier's depot. the distance of a request to a depot is the
    MINIMUM of the distances (delivery - depot) and (depot - delivery)
    """

    def _evaluate_request(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, carrier: slt.AHDSolution,
                          request: int):

        foreign_depots = list(range(instance.num_carriers))
        foreign_depots.pop(carrier.id_)

        dist_min = float('inf')
        delivery = instance.vertex_from_request(request)
        for depot in foreign_depots:
            dist = min(instance.travel_duration([depot], [delivery]),
                       instance.travel_duration([delivery], [depot])).total_seconds()
            if dist < dist_min:
                dist_min = dist

        return dist_min


class MinDistanceToForeignDepotDSum(RequestSelectionBehaviorIndividual):
    """
    Select the requests that are closest to another carrier's depot. the distance of a request to a depot is the
    SUM of the distances (depot - delivery) and (delivery - depot)
    """

    def _evaluate_request(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, carrier: slt.AHDSolution,
                          request: int):

        foreign_depots = list(range(instance.num_carriers))
        foreign_depots.pop(carrier.id_)

        dist_min = float('inf')
        delivery = instance.vertex_from_request(request)

        for depot in foreign_depots:
            dist = sum((instance.travel_distance([depot], [delivery]), instance.travel_distance([delivery], [depot])))
            if dist < dist_min:
                dist_min = dist

        return dist_min


class MinDurationToForeignDepotDSum(RequestSelectionBehaviorIndividual):
    """
    Select the requests that are closest to another carrier's depot. the distance of a request to a depot is the
    SUM of the distances (depot - delivery) and (delivery - depot)
    """

    def _evaluate_request(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, carrier: slt.AHDSolution,
                          request: int):

        foreign_depots = list(range(instance.num_carriers))
        foreign_depots.pop(carrier.id_)

        dur_min = dt.timedelta.max
        delivery = instance.vertex_from_request(request)

        for depot in foreign_depots:
            dur = sum((instance.travel_duration([depot], [delivery]), instance.travel_duration([delivery], [depot])),
                      start=dt.timedelta(0))
            if dur < dur_min:
                dur_min = dur

        return dur_min


class ComboDistRaw(RequestSelectionBehaviorIndividual):
    """
    Combo by Gansterer & Hartl (2016)

    This strategy also combines distance-related and profit-related assessments. A carrier a evaluates a request r
    based on three weighted components: (i) distance to the depot oc of one of the collaborating carriers c (see Fig.
    4), (ii) marginal profit (mprofit), and (iii) distance to the carrier’s own depot oa. Distances are calculated by
    summing up the distances between the pickup node pr and the delivery node dr to the depot (oc or oa).
    """

    def _evaluate_request(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, carrier: slt.AHDSolution,
                          request: int):
        # weighting factors
        alpha1 = 1
        alpha2 = 1
        alpha3 = 1

        # [i] distance to the depot of one of the collaborating carriers
        min_dist_to_foreign_depot = MinDistanceToForeignDepotDSum(None)._evaluate_request(instance, solution,
                                                                                          carrier, request)

        # [ii] marginal profit
        marginal_profit = MarginalProfitProxy(None)._evaluate_request(instance, solution, carrier, request)

        # [iii] distance to the carrier's own depot
        delivery = instance.vertex_from_request(request)
        own_depot = carrier.id_
        # distance as the average of the asymmetric distances between depot and delivery vertex
        dist_to_own_depot = instance.travel_distance([own_depot, delivery], [delivery, own_depot]) / 2

        # weighted sum of non-standardized [i], [ii], [iii]
        valuation = alpha1 * min_dist_to_foreign_depot + alpha2 * marginal_profit - alpha3 * dist_to_own_depot

        return valuation


class ComboDistStandardized(RequestSelectionBehaviorIndividual):
    """
    Combo by Gansterer & Hartl (2016)

    This strategy also combines distance-related and profit-related assessments. A carrier a evaluates a request r
    based on three weighted components: (i) distance to the depot oc of one of the collaborating carriers c (see Fig.
    4), (ii) marginal profit (mprofit), and (iii) distance to the carrier’s own depot oa. Distances are calculated by
    summing up the distances between the pickup node pr and the delivery node dr to the depot (oc or oa).
    """

    def execute(self, instance: it.CAHDInstance, solution: slt.CAHDSolution):
        """
        Overrides method in RequestSelectionBehaviorIndividual because different components of the valuation function
        need to be normalized before summing them up

        select a set of requests based on the concrete selection behavior. will retract the requests from the carrier
        and return

        :return: the auction_request_pool as a list of request indices and a default
        partition, i.e. a list of the carrier indices that maps the auction_request_pool to their original carrier.
        """
        auction_request_pool = []
        original_partition_labels = []

        for carrier in solution.carriers:
            k = _abs_num_requests(carrier, self.num_submitted_requests)
            valuations = []
            for request in carrier.accepted_requests:
                valuation = self._evaluate_request(instance, solution, carrier, request)
                valuations.append(valuation)

            # normalize the valuation components
            standardized_components = []
            for component_series in zip(*valuations):
                mean = sum(component_series) / len(component_series)
                std = sqrt(sum([(x - mean) ** 2 for x in component_series]) / len(component_series))
                standardized_components.append(((x - mean) / std for x in component_series))

            # weighting factors for the three components (min_dist_to_foreign_depot, marginal_profit, dist_to_own_depot)
            # of the valuation function
            alphas = np.array([1, 1, 1])

            # compute the weighted sum of the standardized components
            valuations = []
            for comp in zip(*standardized_components):
                weighted_valuation = alphas * comp
                valuations.append(weighted_valuation)

            # pick the WORST k evaluated requests (from ascending order)
            selected = [r for _, r in sorted(zip(valuations, carrier.accepted_requests))][:k]
            selected.sort()

            for request in selected:
                solution.free_requests_from_carriers(instance, [request])
                auction_request_pool.append(request)
                original_partition_labels.append(carrier.id_)

        return auction_request_pool, original_partition_labels

    def _evaluate_request(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, carrier: slt.AHDSolution,
                          request: int) -> Tuple[float, float, float]:

        # [i] distance to the depot of one of the collaborating carriers
        min_dist_to_foreign_depot = MinDistanceToForeignDepotDSum(None)._evaluate_request(instance, solution,
                                                                                          carrier, request)

        # [ii] marginal profit
        marginal_profit = MarginalProfitProxy(None)._evaluate_request(instance, solution, carrier, request)

        # [iii] distance to the carrier's own depot
        delivery = instance.vertex_from_request(request)
        own_depot = carrier.id_
        # distance as the average of the asymmetric distances between depot and delivery vertex
        dist_to_own_depot = instance.travel_distance([own_depot, delivery], [delivery, own_depot]) / 2

        return min_dist_to_foreign_depot, marginal_profit, dist_to_own_depot


class DepotDurations(RequestSelectionBehaviorIndividual):
    """
    considers min_dur_to_foreign_depot and dur_to_own_depot. Does NOT consider marginal profit!
    evaluates requests as min_dur_to_foreign_depot/dur_to_own_depot
    """

    def _evaluate_request(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, carrier: slt.AHDSolution,
                          request: int) -> Tuple[float, float]:

        # [i] duration to the depot of one of the collaborating carriers
        min_dur_to_foreign_depot = MinDurationToForeignDepotDSum(None)._evaluate_request(instance, solution,
                                                                                         carrier, request)

        # [iii] duration to the carrier's own depot
        vertex = instance.vertex_from_request(request)
        own_depot = carrier.id_
        # duration as the average of the asymmetric durations between depot and delivery vertex
        dur_to_own_depot = instance.travel_duration([own_depot, vertex], [vertex, own_depot]) / 2

        return min_dur_to_foreign_depot.total_seconds()/dur_to_own_depot.total_seconds()


class SpatioTemporal(RequestSelectionBehaviorIndividual):
    """
    a request is more likely to be selected (i.e., submitted to the auction) if it is far away from the depot
    and if its current max_shift value is large
    """

    def _evaluate_request(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, carrier: slt.AHDSolution,
                          request: int):
        vertex = instance.vertex_from_request(request)
        max_dur = max(instance._travel_duration_matrix[carrier.id_])
        dur = instance.travel_duration([carrier.id_], [vertex])
        tour = solution.tour_of_request(request)
        pos = tour.vertex_pos[vertex]
        max_shift = tour.max_shift_sequence[pos]
        max_max_shift = instance.tw_close[vertex] - instance.tw_open[vertex]
        value = dur / max_dur + max_shift / max_max_shift
        if value == 0:
            return float('inf')
        else:
            return 1 / value