import logging
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import final, List, Tuple

from src.cr_ahd.core_module import instance as it, solution as slt, tour as tr
from src.cr_ahd.utility_module import profiling as pr
from src.cr_ahd.routing_module import tour_construction as cns

logger = logging.getLogger(__name__)


class Neighborhood(ABC):
    @abstractmethod
    def feasible_move_generator(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        """
        :return: generator that returns the next found move upon calling the next() method on it. Each move is a
        tuple containing all necessary information to see whether to accept that move and the information to execute
        a move. The first element of that tuple is always the delta in travel distance.
        """
        pass

    @final
    def execute_move(self, instance: it.PDPInstance, solution: slt.CAHDSolution, move: tuple):
        """
        Apply the neighborhood move and register it in the solution's dictionary which keeps track of how often each
        neighborhood operator found a move that was accepted.

        :param instance:
        :param solution:
        :param move: tuple containing all necessary information to execute a move. The first element of that tuple is
         always the delta in travel distance. the remaining ones are e.g. current positions and new insertion positions.
        :return:
        """
        self._execute_move(instance, solution, move)
        self.register_move_execution(solution)
        pass

    @abstractmethod
    def _execute_move(self, instance: it.PDPInstance, solution: slt.CAHDSolution, move: tuple):
        pass

    @abstractmethod
    def feasibility_check(self, instance: it.PDPInstance, solution: slt.CAHDSolution, move: tuple):
        """
        A (hopefully efficient) feasibility check of the neighborhood move. Optimally, this is a constant time
        algorithm. However, the Precedence and Time Window constraints of the PDPTW make efficient move evaluation
        a challenge

        :param instance:
        :param solution:
        :param move:
        :return:
        """
        pass

    @final
    def register_move_execution(self, solution: slt.CAHDSolution):
        """
        update the count of each executed move to see which neighborhoods are more 'successful'.

        :param solution:
        :return:
        """
        solution.local_search_move_counter[self.__class__.__name__] += 1


# =====================================================================================================================
# INTRA-TOUR LOCAL SEARCH
# =====================================================================================================================
class IntraTourNeighborhood(Neighborhood, ABC):
    @final
    def feasible_move_generator(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        for tour in range(solution.carriers[carrier].num_tours()):
            tour_ = solution.carriers[carrier].tours[tour]
            yield from self.feasible_move_generator_for_tour(instance, solution, tour_)  # delegated generator

    @abstractmethod
    def feasible_move_generator_for_tour(self, instance: it.PDPInstance, solution: slt.CAHDSolution, tour_: tr.Tour):
        """

        :param instance:
        :param solution:
        :param tour_:
        :return:
        """
        pass

    @abstractmethod
    def _execute_move(self, instance: it.PDPInstance, solution: slt.CAHDSolution, move: tuple):
        pass

    @abstractmethod
    def feasibility_check(self, instance: it.PDPInstance, solution: slt.CAHDSolution, move):
        pass

    @final
    def first_feasible_move_for_tour(self, instance: it.PDPInstance, solution: slt.CAHDSolution, tour_: tr.Tour):
        raise NotImplementedError
        pass

    @final
    def best_feasible_move_for_tour(self, instance: it.PDPInstance, solution: slt.CAHDSolution, tour_: tr.Tour):
        raise NotImplementedError
        pass


class PDPMove(IntraTourNeighborhood):
    """
    Take a PD pair and see whether inserting it in a different location of the SAME route improves the solution
    """

    def feasible_move_generator_for_tour(self, instance: it.PDPInstance, solution: slt.CAHDSolution, tour_: tr.Tour):
        tour_copy_ = deepcopy(tour_)
        # test all requests
        for old_pickup_pos in range(1, len(tour_copy_) - 2):
            vertex = tour_copy_.routing_sequence[old_pickup_pos]

            # skip if its a delivery vertex
            if instance.vertex_type(vertex) == "delivery":
                continue

            pickup, delivery = instance.pickup_delivery_pair(instance.request_from_vertex(vertex))
            old_delivery_pos = solution.vertex_position_in_tour[delivery]

            delta = 0

            # savings of removing the pickup and delivery
            delta += tour_copy_.pop_distance_delta(instance, (old_pickup_pos, old_delivery_pos))

            # pop
            tour_copy_.pop_and_update(instance, solution, (old_pickup_pos, old_delivery_pos))

            # check all possible new insertions for pickup and delivery vertex of the request
            for new_pickup_pos in range(1, len(tour_copy_)):
                for new_delivery_pos in range(new_pickup_pos + 1, len(tour_copy_) + 1):
                    if new_pickup_pos == old_pickup_pos and new_delivery_pos == old_delivery_pos:
                        continue

                    # cost for inserting request vertices in the new positions
                    delta += tour_copy_.insert_distance_delta(instance, [new_pickup_pos, new_delivery_pos],
                                                              [pickup, delivery])

                    move = (delta, tour_copy_, old_pickup_pos, old_delivery_pos, pickup, delivery, new_pickup_pos,
                            new_delivery_pos)

                    # yield move (with the original tour_, not the copy) if it's feasible
                    if self.feasibility_check(instance, solution, move):
                        move = (delta, tour_, old_pickup_pos, old_delivery_pos, pickup, delivery, new_pickup_pos,
                                new_delivery_pos)
                        yield move

                    # restore delta
                    delta -= tour_copy_.insert_distance_delta(instance, [new_pickup_pos, new_delivery_pos],
                                                              [pickup, delivery])

            # repair
            tour_copy_.insert_and_update(instance, solution, (old_pickup_pos, old_delivery_pos), (pickup, delivery))

    def feasibility_check(self, instance: it.PDPInstance, solution: slt.CAHDSolution, move: tuple):
        delta, tour_, old_pickup_pos, old_delivery_pos, pickup, delivery, new_pickup_pos, new_delivery_pos = move
        assert delivery == pickup + instance.num_requests
        return tour_.insertion_feasibility_check(instance, solution, [new_pickup_pos, new_delivery_pos],
                                                 [pickup, delivery])

    def _execute_move(self, instance: it.PDPInstance, solution: slt.CAHDSolution, move):
        delta, tour_, old_pickup_pos, old_delivery_pos, pickup, delivery, new_pickup_pos, new_delivery_pos = move

        # remove
        pickup_vertex = tour_.routing_sequence[old_pickup_pos]
        delivery_vertex = tour_.routing_sequence[old_delivery_pos]
        tour_.pop_and_update(instance, solution, (old_pickup_pos, old_delivery_pos))

        logger.debug(f'PDPMove: [{delta}] Move vertices {pickup_vertex} and {delivery_vertex} from '
                     f'indices [{old_pickup_pos, old_delivery_pos}] to indices [{new_pickup_pos, new_delivery_pos}]')

        # re-insert
        tour_.insert_and_update(instance, solution, [new_pickup_pos, new_delivery_pos],
                                [pickup_vertex, delivery_vertex])
        pass


class PDPTwoOpt(IntraTourNeighborhood):
    """
    The classic 2-opt neighborhood by Croes (1958)
    """
    def feasible_move_generator_for_tour(self, instance: it.PDPInstance, solution: slt.CAHDSolution, tour_: tr.Tour):
        # iterate over all moves
        for i in range(0, len(tour_) - 3):
            for j in range(i + 2, len(tour_) - 1):

                delta = 0

                # savings of removing the edges (i, i+1) and (j, j+1)
                delta -= instance.distance([tour_.routing_sequence[i], tour_.routing_sequence[j]],
                                           [tour_.routing_sequence[i + 1], tour_.routing_sequence[j + 1]])

                # cost of adding the edges (i, j) and (i+1, j+1)
                delta += instance.distance([tour_.routing_sequence[i], tour_.routing_sequence[i + 1]],
                                           [tour_.routing_sequence[j], tour_.routing_sequence[j + 1]])

                move = (delta, tour_, i, j)

                if self.feasibility_check(instance, solution, move):
                    yield move

    def feasibility_check(self, instance: it.PDPInstance, solution: slt.CAHDSolution, move):
        delta, tour_, i, j = move

        # create a temporary routing sequence to loop over the one that contains the reversed section
        tmp_routing_sequence = list(tour_.routing_sequence)
        tmp_arrival_schedule = list(tour_.arrival_schedule)
        tmp_service_schedule = list(tour_.service_schedule)
        tmp_wait_sequence = list(tour_._wait_sequence)
        tmp_max_shift_sequence = list(tour_._max_shift_sequence)
        pop_indices = list(range(i + 1, j + 1))
        popped, updated_sums = tr.multi_pop_and_update(routing_sequence=tmp_routing_sequence,
                                                       sum_travel_distance=tour_.sum_travel_distance,
                                                       sum_travel_duration=tour_.sum_travel_duration,
                                                       arrival_schedule=tmp_arrival_schedule,
                                                       service_schedule=tmp_service_schedule,
                                                       sum_load=tour_.sum_load,
                                                       sum_revenue=tour_.sum_revenue,
                                                       sum_profit=tour_.sum_profit,
                                                       wait_sequence=tmp_wait_sequence,
                                                       max_shift_sequence=tmp_max_shift_sequence,
                                                       distance_matrix=instance._distance_matrix,
                                                       vertex_load=instance.load,
                                                       revenue=instance.revenue,
                                                       service_duration=instance.service_duration,
                                                       tw_open=solution.tw_open,
                                                       tw_close=solution.tw_close,
                                                       pop_indices=pop_indices)

        return tr.multi_insertion_feasibility_check(
            routing_sequence=tmp_routing_sequence,
            sum_travel_distance=updated_sums['sum_travel_distance'],
            sum_travel_duration=updated_sums['sum_travel_duration'],
            arrival_schedule=tmp_arrival_schedule,
            service_schedule=tmp_service_schedule,
            sum_load=updated_sums['sum_load'],
            sum_revenue=updated_sums['sum_revenue'],
            sum_profit=updated_sums['sum_profit'],
            wait_sequence=tmp_wait_sequence,
            max_shift_sequence=tmp_max_shift_sequence,
            num_depots=instance.num_depots,
            num_requests=instance.num_requests,
            distance_matrix=instance._distance_matrix,
            vertex_load=instance.load,
            revenue=instance.revenue,
            service_duration=instance.service_duration,
            vehicles_max_travel_distance=instance.vehicles_max_travel_distance,
            vehicles_max_load=instance.vehicles_max_load,
            tw_open=solution.tw_open,
            tw_close=solution.tw_close,
            insertion_indices=range(i + 1, j + 1),
            insertion_vertices=list(reversed(popped)),

        )

    def _execute_move(self, instance: it.PDPInstance, solution: slt.CAHDSolution, move):
        delta, tour_, i, j = move

        logger.debug(f'PDPTwoOpt: [{delta}] Reverse section between {i} and {j}')

        tour_.reverse_section(instance, solution, i, j)
        pass


# =====================================================================================================================
# INTER-TOUR LOCAL SEARCH
# =====================================================================================================================
class InterTourNeighborhood(Neighborhood, ABC):
    @abstractmethod
    def feasible_move_generator(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        """
        :return: tuple containing all necessary information to see whether to accept a move and the information to
        execute a move. The first element must be the delta in travel distance.
        """
        pass

    @abstractmethod
    def _execute_move(self, instance: it.PDPInstance, solution: slt.CAHDSolution, move: tuple):
        pass

    @abstractmethod
    def feasibility_check(self, instance: it.PDPInstance, solution: slt.CAHDSolution, move):
        pass


class PDPRelocate(InterTourNeighborhood):
    """
    Take one PD request at a time and see whether inserting it into another tour is cheaper.
    """

    def feasible_move_generator(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        solution = deepcopy(solution)
        carrier_ = solution.carriers[carrier]

        for old_tour in range(carrier_.num_tours()):

            old_tour_ = carrier_.tours[old_tour]

            # check all requests of the old tour
            # a pickup can be at most at index n-2 (n is depot, n-1 must be delivery)
            for old_pickup_pos in range(1, len(old_tour_) - 2):
                vertex = old_tour_.routing_sequence[old_pickup_pos]

                # skip if its a delivery vertex
                if instance.vertex_type(vertex) == "delivery":
                    continue  # skip if its a delivery vertex

                pickup, delivery = instance.pickup_delivery_pair(instance.request_from_vertex(vertex))
                old_delivery_pos = old_tour_.routing_sequence.index(delivery)

                # check all new tours for re-insertion
                for new_tour in range(carrier_.num_tours()):
                    new_tour_: tr.Tour = carrier_.tours[new_tour]

                    # skip the current tour, there is the PDPMove local search for that option
                    if new_tour == old_tour:
                        continue

                    # check all possible new insertions for pickup and delivery vertex of the request
                    for new_pickup_pos in range(1, len(new_tour_)):
                        for new_delivery_pos in range(new_pickup_pos + 1, len(new_tour_) + 1):

                            delta = 0
                            # savings of removing the pickup and delivery
                            delta += old_tour_.pop_distance_delta(instance, (old_pickup_pos, old_delivery_pos))
                            # cost for inserting request vertices in the new positions
                            delta += new_tour_.insert_distance_delta(instance,
                                                                     [new_pickup_pos, new_delivery_pos],
                                                                     [pickup, delivery])

                            move = (
                                delta, carrier, old_tour_, old_pickup_pos, old_delivery_pos, new_tour_, new_pickup_pos,
                                new_delivery_pos)

                            # is the improving move feasible?
                            if self.feasibility_check(instance, solution, move):
                                yield move

    def feasibility_check(self, instance: it.PDPInstance, solution: slt.CAHDSolution, move):
        delta, carrier, old_tour_, old_pickup_pos, old_delivery_pos, new_tour_, new_pickup_pos, new_delivery_pos = move
        pickup = old_tour_.routing_sequence[old_pickup_pos]
        delivery = old_tour_.routing_sequence[old_delivery_pos]

        return new_tour_.insertion_feasibility_check(instance, solution, [new_pickup_pos, new_delivery_pos],
                                                     [pickup, delivery])

    def _execute_move(self, instance: it.PDPInstance, solution: slt.CAHDSolution, move):
        delta, carrier, old_tour_, old_pickup_pos, old_delivery_pos, new_tour_, new_pickup_pos, new_delivery_pos = move

        pickup, delivery = old_tour_.pop_and_update(instance, solution, [old_pickup_pos, old_delivery_pos])

        logger.debug(f'PDPRelocate: [{delta}] Relocate vertices {pickup} and {delivery} from '
                     f'Tour {old_tour_.id_} [{old_pickup_pos, old_delivery_pos}] to '
                     f'Tour {new_tour_.id_} [{new_pickup_pos, new_delivery_pos}]')

        # if it is now empty (i.e. depot -> depot), drop the old tour
        if len(old_tour_) <= 2:
            solution.carriers[carrier].tours.execute(old_tour_)

        new_tour_.insert_and_update(instance, solution, [new_pickup_pos, new_delivery_pos], [pickup, delivery])
        solution.request_to_tour_assignment[instance.request_from_vertex(pickup)] = new_tour_.id_

        pass


'''
class PDPRelocate2(InterTourLocalSearchBehavior):
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


class PDPMergeTours(InterTourNeighborhood):
    """take all requests of a tour and see whether inserting them into some other tours improves the solution"""
    pass
