import datetime as dt
from abc import ABC, abstractmethod

from auction_module.request_selection.individual import MarginalProfitProxy
from auction_module.request_selection.request_selection import RequestSelectionBehavior, _abs_num_requests
from core_module import instance as it, solution as slt


class RequestSelectionBehaviorNeighbor(RequestSelectionBehavior, ABC):
    """
    Select (for each carrier) a bundle by finding an initial request and then adding num_submitted_requests-1 more
    requests based on some neighborhood criterion, idea from
    Gansterer, Margaretha, and Richard F. Hartl. 2016. “Request Evaluation Strategies for Carriers in Auction-Based
    Collaborations.” OR Spectrum 38 (1): 3–23. https://doi.org/10.1007/s00291-015-0411-1.
    """

    def execute(self, instance: it.CAHDInstance, solution: slt.CAHDSolution):
        auction_request_pool = []
        original_partition_labels = []

        for carrier in solution.carriers:
            k = _abs_num_requests(carrier, self.num_submitted_requests)

            # find an initial request
            initial_request = self._find_initial_request(instance, solution, carrier)

            # find the best k-1 neighboring ones
            neighbors = self._find_neighbors(instance, solution, carrier, initial_request, k - 1)

            best_bundle = [initial_request] + neighbors
            best_bundle.sort()

            # carrier's best bundles: retract requests from their tours and add them to auction pool & original partition
            for request in best_bundle:
                solution.free_requests_from_carriers(instance, [request])

                # update auction pool and original partition candidate
                auction_request_pool.append(request)
                original_partition_labels.append(carrier.id_)

        return auction_request_pool, original_partition_labels

    @abstractmethod
    def _find_initial_request(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, carrier: slt.AHDSolution):
        pass

    def _find_neighbors(self,
                        instance: it.CAHDInstance, solution: slt.CAHDSolution,
                        carrier_id: int, initial_request: int,
                        num_neighbors: int):
        """
        "Any further request s ∈ Ra is selected based on its closeness to r. Closeness is determined by the sum of
        distances between four nodes (pickup nodes [pr, ps] and delivery nodes [dr, ds])."

        In VRP closeness is determined as the sum of the distances: dist(r, s) + dist(s,r)

        :param solution:
        :param instance:
        :param carrier_id:
        :param initial_request: r
        :param num_neighbors:
        :return:
        """
        carrier = solution.carriers[carrier_id]
        init_delivery = instance.vertex_from_request(initial_request)

        neighbor_valuations = []
        for neighbor in carrier.accepted_requests:
            if neighbor == initial_request:
                neighbor_valuations.append(float('inf'))

            neigh_delivery = instance.vertex_from_request(neighbor)
            valuation = instance.travel_distance([init_delivery, neigh_delivery], [neigh_delivery, init_delivery])
            neighbor_valuations.append(valuation)

        # sort by valuations
        best_neigh = [neigh for val, neigh in sorted(zip(neighbor_valuations, carrier.accepted_requests))]

        return best_neigh[:num_neighbors]


class MarginalProfitProxyNeighbor(RequestSelectionBehaviorNeighbor):
    """
    The first request r ∈ Ra is selected using MarginalProfitProxy. Any further request s ∈ Ra is selected based on
    its closeness to r.
    """

    def _find_initial_request(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, carrier: slt.AHDSolution):
        min_marginal_profit = float('inf')
        initial_request = None
        for request in carrier.accepted_requests:
            marginal_profit = MarginalProfitProxy(1)._evaluate_request(instance, solution, carrier, request)
            if marginal_profit < min_marginal_profit:
                min_marginal_profit = marginal_profit
                initial_request = request

        return initial_request


class SuccessorsNeighbor(RequestSelectionBehaviorNeighbor):
    """
    the initial request is the one with the longest waiting time. neighbors are any succeeding requests. If there
    are not sufficient successors, the requests that have the shortest travel duration from the initial request
    will be chosen.
    """

    def _find_initial_request(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, carrier: slt.AHDSolution):
        max_wait = dt.timedelta(0)
        max_wait_vertex = None
        for tour in carrier.tours:
            wait, vertex = max(zip(tour.wait_duration_sequence, tour.routing_sequence))
            if wait > max_wait:
                max_wait = wait
                max_wait_vertex = vertex
        return instance.request_from_vertex(max_wait_vertex)

    def _find_neighbors(self,
                        instance: it.CAHDInstance, solution: slt.CAHDSolution,
                        carrier_id: int, initial_request: int,
                        num_neighbors: int):
        tour = solution.tour_of_request(initial_request)
        pos = tour.vertex_pos[instance.vertex_from_request(initial_request)]
        num_successors = len(tour.routing_sequence[pos + 1:-1])
        if num_successors >= num_neighbors:
            return [instance.request_from_vertex(v) for v in tour.routing_sequence[pos + 1:pos + 1 + num_neighbors]]
        else:
            neighbors = [instance.request_from_vertex(v) for v in tour.routing_sequence[pos + 1:-1]]
            carrier = solution.carriers[carrier_id]
            initial_vertex = instance.vertex_from_request(initial_request)
            nearest_neighbors = sorted(carrier.accepted_requests,
                                       key=lambda x: instance.travel_duration(
                                           [initial_vertex], [instance.vertex_from_request(x)]))
            while len(neighbors) < num_neighbors:
                candidate = next(nearest_neighbors)
                if candidate not in [initial_vertex] + neighbors:
                    neighbors.append(candidate)
            return neighbors