import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import final

from src.cr_ahd.core_module import instance as it, solution as slt, tour as tr

logger = logging.getLogger(__name__)


class Neighborhood(ABC):
    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def feasible_move_generator_for_carrier(self, instance: it.MDPDPTWInstance, carrier: slt.AHDSolution):
        """
        :return: generator that returns the next found move upon calling the next() method on it. Each move is a
        tuple containing all necessary information to see whether to accept that move and the information to execute
        a move. The first element of that tuple is always the delta *IN TRAVEL DISTANCE*.
        """
        pass

    @abstractmethod
    def execute_move(self, instance: it.MDPDPTWInstance, move: tuple):
        """
        Executes the neighbourhood move in place.

        :param move: tuple containing all necessary information to execute a move. The first element of that tuple is
         always the delta in travel distance. the remaining ones are e.g. current positions and new insertion positions.
        """
        pass

    @abstractmethod
    def feasibility_check(self, instance: it.MDPDPTWInstance, move: tuple):
        """
        A (hopefully efficient) feasibility check of the neighborhood move. Optimally, this is a constant time
        algorithm. However, the Precedence and Time Window constraints of the PDPTW make efficient move evaluation
        a challenge

        :param instance:
        :param move:
        :return:
        """
        pass


# =====================================================================================================================
# INTRA-TOUR NEIGHBORHOOD
# =====================================================================================================================
class IntraTourNeighborhood(Neighborhood, ABC):
    @final
    def feasible_move_generator_for_carrier(self, instance: it.MDPDPTWInstance, carrier: slt.AHDSolution):
        """
        using a generator (i.e. "yield" instead of "return") avoids unnecessary move evaluations. E.g., in a first
        improvement scenario, it is not necessary to evaluate (or even define) the complete neighborhood, finding a
        single, feasible and improving move is sufficient here.

        :param instance:
        :param carrier:
        :return:
        """
        for tour in carrier.tours:
            yield from self.feasible_move_generator_for_tour(instance, tour)  # delegated generator

    @abstractmethod
    def feasible_move_generator_for_tour(self, instance: it.MDPDPTWInstance, tour: tr.Tour):
        """

        :param instance:
        :param tour:
        :return:
        """
        pass

    @abstractmethod
    def feasibility_check(self, instance: it.MDPDPTWInstance, move):
        pass

    @final
    def first_feasible_move_for_tour(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution, tour_: tr.Tour):
        raise NotImplementedError
        pass

    @final
    def best_feasible_move_for_tour(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution, tour_: tr.Tour):
        raise NotImplementedError
        pass


class PDPMove(IntraTourNeighborhood):
    """
    Take a PD pair and see whether inserting it in a different location of the SAME route improves the solution.
    move = (delta, tour_, old_pickup_pos, old_delivery_pos, pickup, delivery, new_pickup_pos, new_delivery_pos)
    """

    def feasible_move_generator_for_tour(self, instance: it.MDPDPTWInstance, tour: tr.Tour):
        tour_copy = deepcopy(tour)
        # test all requests
        for old_pickup_pos in range(1, len(tour_copy) - 2):
            vertex = tour_copy.routing_sequence[old_pickup_pos]

            # skip if its a delivery vertex
            if instance.vertex_type(vertex) == "delivery":
                continue

            pickup, delivery = instance.pickup_delivery_pair(instance.request_from_vertex(vertex))
            old_delivery_pos = tour_copy.vertex_pos[delivery]

            delta = 0

            # savings of removing the pickup and delivery
            delta += tour_copy.pop_distance_delta(instance, (old_pickup_pos, old_delivery_pos))

            # pop
            tour_copy.pop_and_update(instance, (old_pickup_pos, old_delivery_pos))

            # check all possible new insertions for pickup and delivery vertex of the request
            for new_pickup_pos in range(1, len(tour_copy)):
                for new_delivery_pos in range(new_pickup_pos + 1, len(tour_copy) + 1):
                    if new_pickup_pos == old_pickup_pos and new_delivery_pos == old_delivery_pos:
                        continue

                    # cost for inserting request vertices in the new positions
                    insertion_distance_delta = tour_copy.insert_distance_delta(instance,
                                                                                [new_pickup_pos, new_delivery_pos],
                                                                                [pickup, delivery])
                    delta += insertion_distance_delta

                    move = (delta, tour_copy, old_pickup_pos, old_delivery_pos, pickup, delivery, new_pickup_pos,
                            new_delivery_pos)

                    # yield move (with the original tour_, not the copy) if it's feasible
                    if self.feasibility_check(instance, move):
                        move = (delta, tour, old_pickup_pos, old_delivery_pos, pickup, delivery, new_pickup_pos,
                                new_delivery_pos)
                        yield move

                    # restore delta before checking the next insertion positions
                    delta -= insertion_distance_delta

            # repair the copy before checking the next request for repositioning
            tour_copy.insert_and_update(instance, (old_pickup_pos, old_delivery_pos), (pickup, delivery))

    def feasibility_check(self, instance: it.MDPDPTWInstance, move: tuple):
        delta, tour, old_pickup_pos, old_delivery_pos, pickup, delivery, new_pickup_pos, new_delivery_pos = move
        assert delivery == pickup + instance.num_requests
        return tour.insertion_feasibility_check(instance, [new_pickup_pos, new_delivery_pos], [pickup, delivery])

    def execute_move(self, instance: it.MDPDPTWInstance, move):
        delta, tour, old_pickup_pos, old_delivery_pos, pickup, delivery, new_pickup_pos, new_delivery_pos = move

        # remove
        pickup_vertex = tour.routing_sequence[old_pickup_pos]
        delivery_vertex = tour.routing_sequence[old_delivery_pos]
        tour.pop_and_update(instance, (old_pickup_pos, old_delivery_pos))

        logger.debug(f'PDPMove: [{delta}] Move vertices {pickup_vertex} and {delivery_vertex} from '
                     f'indices [{old_pickup_pos, old_delivery_pos}] to indices [{new_pickup_pos, new_delivery_pos}]')

        # re-insert
        tour.insert_and_update(instance, [new_pickup_pos, new_delivery_pos], [pickup_vertex, delivery_vertex])
        pass


class PDPTwoOpt(IntraTourNeighborhood):
    """
    The classic 2-opt neighborhood by Croes (1958)
    """

    def feasible_move_generator_for_tour(self, instance: it.MDPDPTWInstance, tour: tr.Tour):
        # iterate over all moves
        for i in range(0, len(tour) - 3):
            for j in range(i + 2, len(tour) - 1):
                # computing the distance delta assumes symmetric distances
                delta = 0

                # savings of removing the edges (i, i+1) and (j, j+1)
                delta -= instance.distance([tour.routing_sequence[i], tour.routing_sequence[j]],
                                           [tour.routing_sequence[i + 1], tour.routing_sequence[j + 1]])

                # cost of adding the edges (i, j) and (i+1, j+1)
                delta += instance.distance([tour.routing_sequence[i], tour.routing_sequence[i + 1]],
                                           [tour.routing_sequence[j], tour.routing_sequence[j + 1]])

                move = (delta, tour, i, j)

                if self.feasibility_check(instance, move):
                    yield move

    def feasibility_check(self, instance: it.MDPDPTWInstance, move):
        tour: tr.Tour
        delta, tour, i, j = move

        # create a copy to check feasibility
        pop_indices = list(range(i + 1, j + 1))
        popped = tour.pop_and_update(instance, pop_indices)
        feasibility = tour.insertion_feasibility_check(instance, range(i + 1, j + 1), list(reversed(popped)))
        tour.insert_and_update(instance, pop_indices, popped)
        return feasibility

    def execute_move(self, instance: it.MDPDPTWInstance, move):
        delta, tour, i, j = move

        logger.debug(f'PDPTwoOpt: [{delta}] Reverse section between {i} and {j}')

        popped = tour.pop_and_update(instance, list(range(i + 1, j + 1)))
        tour.insert_and_update(instance, range(i + 1, j + 1), list(reversed(popped)))
        pass


# =====================================================================================================================
# INTER-TOUR NEIGHBORHOOD
# =====================================================================================================================
class InterTourNeighborhood(Neighborhood, ABC):
    @abstractmethod
    def feasible_move_generator_for_carrier(self, instance: it.MDPDPTWInstance, carrier: slt.AHDSolution):
        """
        :return: tuple containing all necessary information to see whether to accept a move and the information to
        execute a move. The first element must be the delta in travel distance.
        """
        pass

    @abstractmethod
    def feasibility_check(self, instance: it.MDPDPTWInstance, move):
        pass


class PDPRelocate(InterTourNeighborhood):
    """
    Take one PD request at a time and see whether inserting it into another tour is cheaper.
    """

    def feasible_move_generator_for_carrier(self, instance: it.MDPDPTWInstance, carrier: slt.AHDSolution):
        # raise NotImplementedError

        for old_tour in carrier.tours:

            # check all requests of the old tour
            for old_pickup_pos in range(1, len(old_tour) - 2):
                vertex = old_tour.routing_sequence[old_pickup_pos]

                # skip if its a delivery vertex
                if instance.vertex_type(vertex) == "delivery":
                    continue  # skip if its a delivery vertex

                pickup, delivery = instance.pickup_delivery_pair(instance.request_from_vertex(vertex))
                old_delivery_pos = old_tour.vertex_pos[delivery]

                # check all new tours for re-insertion
                for new_tour in carrier.tours:

                    # skip the current tour, there is PDPMove for that option
                    if new_tour is old_tour:
                        continue

                    # check all possible new insertions for pickup and delivery vertex of the request
                    for new_pickup_pos in range(1, len(new_tour)):
                        for new_delivery_pos in range(new_pickup_pos + 1, len(new_tour) + 1):

                            delta = 0
                            # savings of removing the pickup and delivery
                            delta += old_tour.pop_distance_delta(instance, (old_pickup_pos, old_delivery_pos))
                            # cost for inserting request vertices in the new positions
                            delta += new_tour.insert_distance_delta(instance,
                                                                     [new_pickup_pos, new_delivery_pos],
                                                                     [pickup, delivery])

                            move = (
                                delta, carrier, old_tour, old_pickup_pos, old_delivery_pos, new_tour, new_pickup_pos,
                                new_delivery_pos)

                            # is the improving move feasible?
                            if self.feasibility_check(instance, move):
                                yield move

    def feasibility_check(self, instance: it.MDPDPTWInstance, move):
        delta, carrier, old_tour, old_pickup_pos, old_delivery_pos, new_tour, new_pickup_pos, new_delivery_pos = move
        pickup = old_tour.routing_sequence[old_pickup_pos]
        delivery = old_tour.routing_sequence[old_delivery_pos]
        return new_tour.insertion_feasibility_check(instance, [new_pickup_pos, new_delivery_pos], [pickup, delivery])

    def execute_move(self, instance: it.MDPDPTWInstance, move):
        delta, carrier, old_tour, old_pickup_pos, old_delivery_pos, new_tour, new_pickup_pos, new_delivery_pos = move

        pickup, delivery = old_tour.pop_and_update(instance, [old_pickup_pos, old_delivery_pos])
        request = instance.request_from_vertex(pickup)
        old_tour.requests.remove(request)
        logger.debug(f'PDPRelocate: [{delta}] Relocate vertices {pickup} and {delivery} from '
                     f'Tour {old_tour.id_} {old_pickup_pos, old_delivery_pos} to '
                     f'Tour {new_tour.id_} {new_pickup_pos, new_delivery_pos}')

        # if it is now empty (i.e. depot -> depot), drop the old tour
        if len(old_tour) <= 2:
            carrier.tours.remove(old_tour)

        new_tour.insert_and_update(instance, [new_pickup_pos, new_delivery_pos], [pickup, delivery])
        new_tour.requests.add(request)
        pass


'''
class PDPRelocate2(InterTourLocalSearchBehavior):
    """
    relocate two requests at once
    """
    def feasible_move_generator(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        solution = deepcopy(solution)
        carrier_ = solution.carriers[carrier]

        for old_tour in range(carrier_.num_tours()):

            old_tour_ = carrier_.tours[old_tour]

            # check all requests of the old tour
            # a pickup_1 can be at most at index n-2 (n is depot, n-1 must be delivery_1)
            for old_pickup_pos_1 in range(1, len(old_tour_) - 2):
                vertex_1 = old_tour_.routing_sequence[old_pickup_pos_1]

                # skip if its a delivery_1 vertex_1
                if instance.vertex_type(vertex_1) == "delivery":
                    continue  # skip if its a delivery_1 vertex_1

                pickup_1, delivery_1 = instance.pickup_delivery_pair(instance.request_from_vertex(vertex_1))
                old_delivery_pos_1 = old_tour_.routing_sequence.index(delivery_1)

                for old_pickup_pos_2 in range(old_pickup_pos_1 + 1, len(old_tour_) - 2):
                    vertex_2 = old_tour_.routing_sequence[old_pickup_pos_2]

                    # skip if its a delivery_1 vertex_1
                    if instance.vertex_type(vertex_2) == "delivery":
                        continue  # skip if its a delivery_2 vertex_2

                    pickup_2, delivery_2 = instance.pickup_delivery_pair(instance.request_from_vertex(vertex_2))
                    old_delivery_pos_2 = old_tour_.routing_sequence.index(delivery_2)

                    # check all new tours for re-insertion
                    for new_tour_1 in range(carrier_.num_tours()):
                        new_tour_1_: tr.Tour = carrier_.tours[new_tour_1]

                        # skip the current tour, there is the PDPMove local search for that option
                        if new_tour_1 == old_tour:
                            continue

                        # check all possible new insertions for pickup_1 and delivery_1
                        for new_pickup_pos_1 in range(1, len(new_tour_1_)):
                            for new_delivery_pos_1 in range(new_pickup_pos_1 + 1, len(new_tour_1_) + 1):

                                # check first relocation and execute if feasible
                                if new_tour_1_.insertion_feasibility_check(instance, solution,
                                                                           [new_pickup_pos_1, new_delivery_pos_1],
                                                                           [pickup_1, delivery_1]):
                                    new_tour_1_.insert_and_update(instance, solution,
                                                                  [new_pickup_pos_1, new_delivery_pos_1],
                                                                  [pickup_1, delivery_1])
                                else:
                                    continue

                                for new_tour_2 in range(carrier_.num_tours()):
                                    new_tour_2_: tr.Tour = carrier_.tours[new_tour_2]
                                    if new_tour_2 == old_tour:
                                        continue
                                    for new_pickup_pos_2 in range(1, len(new_tour_2_)):
                                        for new_delivery_pos_2 in range(new_pickup_pos_2 + 1, len(new_tour_2_) + 1):

                                            delta = 0
                                            # savings of removing the pickup_1, delivery_1, pickup_2 & delivery_2
                                            delta += old_tour_.pop_distance_delta(instance, sorted(
                                                (old_pickup_pos_1, old_delivery_pos_1, old_pickup_pos_2,
                                                 old_delivery_pos_2)))
                                            # cost for inserting them at their potential new positions
                                            delta += new_tour_1_.insert_distance_delta(instance,
                                                                                       [new_pickup_pos_1,
                                                                                        new_delivery_pos_1],
                                                                                       [pickup_1, delivery_1])
                                            delta += new_tour_2_.insert_distance_delta(instance,
                                                                                       [new_pickup_pos_2,
                                                                                        new_delivery_pos_2],
                                                                                       [pickup_2, delivery_2])

                                            if new_tour_2_.insertion_feasibility_check(instance, solution,
                                                                                       [new_pickup_pos_2,
                                                                                        new_delivery_pos_2],
                                                                                       [pickup_2, delivery_2]):
                                                move = (
                                                    delta, old_tour, old_pickup_pos_1, old_delivery_pos_1, new_tour_1,
                                                    new_pickup_pos_1, new_delivery_pos_1, old_pickup_pos_2,
                                                    old_delivery_pos_2, new_tour_2, new_pickup_pos_2,
                                                    new_delivery_pos_2)

                                                yield move
                                            else:
                                                continue

                                # restore the initial state
                                new_tour_1_.pop_and_update(instance, solution, [new_pickup_pos_1, new_delivery_pos_1])

    def _execute_move(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, move):
        delta, old_tour, old_pickup_pos_1, old_delivery_pos_1, new_tour_1, new_pickup_pos_1, new_delivery_pos_1, \
        old_pickup_pos_2, old_delivery_pos_2, new_tour_2, new_pickup_pos_2, new_delivery_pos_2 = move

        old_tour_ = solution.carriers[carrier].tours[old_tour]

        pickup_1 = old_tour_.routing_sequence[old_pickup_pos_1]
        delivery_1 = old_tour_.routing_sequence[old_delivery_pos_1]
        pickup_2 = old_tour_.routing_sequence[old_pickup_pos_2]
        delivery_2 = old_tour_.routing_sequence[old_delivery_pos_2]

        old_tour_.pop_and_update(instance, solution,
                                 sorted((old_pickup_pos_1, old_delivery_pos_1, old_pickup_pos_2, old_delivery_pos_2)))
        solution.carriers[carrier].tours[new_tour_1].insert_and_update(instance, solution,
                                                                       [new_pickup_pos_1, new_delivery_pos_1],
                                                                       [pickup_1, delivery_1])
        solution.carriers[carrier].tours[new_tour_2].insert_and_update(instance, solution,
                                                                       [new_pickup_pos_2, new_delivery_pos_2],
                                                                       [pickup_2, delivery_2])

        pass

    def feasibility_check(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, move):
        """for this local search, the feasibility check is incorporated in the feasible_move_generator"""
        pass

'''
