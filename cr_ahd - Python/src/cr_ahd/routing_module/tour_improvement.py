import logging
from abc import ABC, abstractmethod

from src.cr_ahd.core_module import instance as it, solution as slt

logger = logging.getLogger(__name__)


# TODO implement a best-improvement vs first-improvement mechanism on the parent-class level. e.g. as a self.best:bool
class TourImprovementBehavior(ABC):

    @abstractmethod
    def improve_global_solution(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        pass

    @abstractmethod
    def improve_carrier_solution(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int):
        pass


class TwoOpt(TourImprovementBehavior):
    """
    Improve the current solution with a 2-opt local search as in
    G. A. Croes, A method for routing_module traveling salesman problems. Operations Res. 6 (1958)
    """

    def improve_global_solution(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        for carrier in range(instance.num_carriers):
            self.improve_carrier_solution(instance, solution, carrier)
        pass

    def improve_carrier_solution(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int):
        """Applies the 2-Opt local search operator to all vehicles/tours"""
        for tour in solution.carrier_solutions[carrier].num_tours():
            improved = True
            while improved:
                improved = False
                best_pos_i, best_pos_j, best_delta = self.improve_tour(instance, solution, carrier, tour)
                if best_delta < 0:
                    solution.carrier_solutions[carrier].tours[tour].reverse_section(instance, solution, best_pos_i,
                                                                                    best_pos_j)
                    improved = True
                    logger.debug(f'2Opt: reversing section between positions {best_pos_i} and {best_pos_j}')
        pass

    def improve_tour(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, tour: int,
                     best_improvement=True):
        best_pos_i = None
        best_pos_j = None
        best_delta = float('inf')
        t = solution.carrier_solutions[carrier].tours[tour]
        for i in range(1, len(t) - 2):
            for j in range(i + 1, len(t)):
                if j - i == 1:
                    continue  # no effect
                delta = -instance.distance([i, j - 1], [i + 1, j]) + instance.distance([i, i + 1], [j - 1, j])
                if delta < best_delta:
                    best_pos_i = i
                    best_pos_j = j
                    best_delta = delta
                    if not best_improvement and delta < 0:
                        return best_pos_i, best_pos_j, best_delta
        return best_pos_i, best_pos_j, best_delta


class PDPExchange(TourImprovementBehavior):
    """
    Take one PD request at a time and see whether inserting it into another tour is cheaper.
    BEST improvement for each PD request.
    """

    def improve_global_solution(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        for carrier in range(instance.num_carriers):
            self.improve_carrier_solution(instance, solution, carrier)
        pass

    def improve_carrier_solution(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int):
        cs = solution.carrier_solutions[carrier]
        for old_tour in range(cs.num_tours()):
            # TODO do i want best improvement also on this level? This would mean checking ALL exchanges, i.e. greedily
            #  finding the best request to relocate (with the highest savings) before checking possible relocations.
            #  not for now ...
            best_vertex = None
            best_savings = None

            improved = True
            while improved:
                improved = False
                # a pickup at be at most at index n-2 (n is depot, n-1 must be delivery)
                for pickup_position in range(1, len(cs.tours[old_tour]) - 2):
                    vertex = cs.tours[old_tour].routing_sequence[pickup_position]
                    if vertex >= instance.num_carriers + instance.num_requests:
                        continue  # skip if its a delivery vertex
                    pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(
                        instance.request_from_vertex(vertex))
                    delivery_position = cs.tours[old_tour].routing_sequence.index(delivery_vertex)
                    # positive savings obtained by removing the requests
                    savings = cs.tours[old_tour].removal_distance_delta(instance, [pickup_position, delivery_position])

                    best_delta, best_new_tour, best_pos_i, best_pos_j = self.find_best_new_insertion(instance,
                                                                                                     solution,
                                                                                                     carrier,
                                                                                                     old_tour,
                                                                                                     pickup_vertex,
                                                                                                     delivery_vertex)
                    if best_delta < abs(savings):
                        cs.exchange_vertices(instance, solution, old_tour, best_new_tour,
                                             [pickup_position, delivery_position], [best_pos_i, best_pos_j])
                        improved = True
                        # whenever an exchange has been applied, the tour must be searched from the beginning again
                        break
        pass

    def find_best_new_insertion(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int,
                                old_tour: int, pickup_vertex: int, delivery_vertex: int):
        cs = solution.carrier_solutions[carrier]
        best_delta = float('inf')
        best_new_tour = None
        best_pos_i = None
        best_pos_j = None
        for new_tour in range(cs.num_tours()):
            if new_tour == old_tour:
                continue
            for i in range(1, len(cs.tours[new_tour]) - 1):
                for j in range(i + 1, len(cs.tours[new_tour])):
                    delta = cs.tours[new_tour].insertion_distance_delta(instance, [i, j],
                                                                        [pickup_vertex, delivery_vertex])
                    if delta < best_delta and cs.tours[new_tour].insertion_feasibility_check(instance, solution, [i, j],
                                                                                             [pickup_vertex,
                                                                                              delivery_vertex]):
                        best_delta = delta
                        best_new_tour = new_tour
                        best_pos_i = i
                        best_pos_j = j
        return best_delta, best_new_tour, best_pos_i, best_pos_j


# class ThreeOpt(TourImprovementBehavior):
#     pass

# class LinKernighan(TourImprovementBehavior):
#     pass

# class Swap(TourImprovementBehavior):
#     pass

class NoImprovement(TourImprovementBehavior):
    def improve_global_solution(self, instance):
        pass

    def improve_carrier_solution(self, carrier):
        pass

    def improve_tour(self, tour):
        pass
