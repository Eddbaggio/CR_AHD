import logging
import typing
from abc import ABC, abstractmethod
from copy import deepcopy

from core_module import instance as it, solution as slt, tour as tr

logger = logging.getLogger(__name__)

vrptw_move_move = typing.Tuple[float, tr.Tour, int, int, int]  # delta, tour, old_pos, vertex, new_pos
vrptw_2opt_move = typing.Tuple[float, tr.Tour, int, int]  # (delta, tour, i, j)
vrptw_relocate_move = typing.Tuple[
    float, slt.AHDSolution, tr.Tour, int, tr.Tour, int]  # (delta, carrier, old_tour, old_pos, new_tour, new_pos)


class Neighborhood(ABC):
    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def feasible_move_generator_for_carrier(self, instance: it.CAHDInstance, carrier: slt.AHDSolution):
        """
        :return: generator that returns the next found move upon calling the next() method on it. Each move is a
        tuple containing all necessary information to see whether to accept that move and the information to execute
        a move. The first element of that tuple is always the delta *IN TRAVEL DISTANCE*.
        """
        pass

    @abstractmethod
    def execute_move(self, instance: it.CAHDInstance, move: tuple):
        """
        Executes the neighbourhood move in place.

        :param move: tuple containing all necessary information to execute a move. The first element of that tuple is
         always the delta in travel distance. the remaining ones are e.g. current positions and new insertion positions.
        """
        pass

    @abstractmethod
    def feasibility_check(self, instance: it.CAHDInstance, move: tuple):
        """
        A (hopefully efficient) feasibility check of the neighborhood move. Optimally, this is a constant time
        algorithm. However, the Precedence and Time Window constraints of the PDPTW make efficient move evaluation
        a challenge

        :param instance:
        :param move:
        :return:
        """
        pass


class NoNeighborhood(Neighborhood):
    def feasible_move_generator_for_carrier(self, instance: it.CAHDInstance, carrier: slt.AHDSolution):
        pass

    def execute_move(self, instance: it.CAHDInstance, move: tuple):
        pass

    def feasibility_check(self, instance: it.CAHDInstance, move: tuple):
        pass


# =====================================================================================================================
# INTRA-TOUR NEIGHBORHOOD
# =====================================================================================================================
class IntraTourNeighborhood(Neighborhood, ABC):
    @typing.final
    def feasible_move_generator_for_carrier(self, instance: it.CAHDInstance, carrier: slt.AHDSolution):
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
    def feasible_move_generator_for_tour(self, instance: it.CAHDInstance, tour: tr.Tour):
        """

        :param instance:
        :param tour:
        :return:
        """
        pass

    @abstractmethod
    def feasibility_check(self, instance: it.CAHDInstance, move):
        pass


class VRPTWMoveDist(IntraTourNeighborhood):
    """
    take a delivery vertex and see whether inserting it in a different location of the SAME tour improves the solution.
    move = (delta, tour, old_pos, vertex, new_pos)
    """

    def feasible_move_generator_for_tour(self, instance: it.CAHDInstance, tour: tr.Tour):
        tour_copy = deepcopy(tour)
        for old_pos, vertex in enumerate(tour_copy.routing_sequence[1:-1], start=1):
            delta = tour_copy.pop_distance_delta(instance, [old_pos])
            tour_copy.pop_and_update(instance, [old_pos])
            # check new insertion positions
            for new_pos in range(1, len(tour_copy) - 1):
                if new_pos == old_pos:
                    continue
                insertion_distance_delta = tour_copy.insert_distance_delta(instance, [new_pos], [vertex])
                delta += insertion_distance_delta
                move = (delta, tour_copy, old_pos, vertex, new_pos)
                # yield the move (with the original tour, not the copy) if it's feasible
                if self.feasibility_check(instance, move):
                    move = (delta, tour, old_pos, vertex, new_pos)
                    yield move
                # restore delta before checking the next insertion position
                delta -= insertion_distance_delta
            # repair the copy before checking the next request for repositioning
            tour_copy.insert_and_update(instance, [old_pos], [vertex])
        pass

    def feasibility_check(self, instance: it.CAHDInstance, move: vrptw_move_move):
        delta, tour, old_pos, vertex, new_pos = move
        return tour.insertion_feasibility_check(instance, [new_pos], [vertex])

    def execute_move(self, instance: it.CAHDInstance, move: vrptw_move_move):
        delta, tour, old_pos, vertex, new_pos = move
        # remove
        tour.pop_and_update(instance, [old_pos])
        tour.insert_and_update(instance, [new_pos], [vertex])
        logger.debug(
            f'{self.name} neighborhood moved vertex {vertex} from {old_pos} to {new_pos} for a delta of {delta}')
        pass


class VRPTWMoveDur(VRPTWMoveDist):
    def feasible_move_generator_for_tour(self, instance: it.CAHDInstance, tour: tr.Tour):
        tour_copy = deepcopy(tour)
        for old_pos, vertex in enumerate(tour_copy.routing_sequence[1:-1], start=1):
            delta = tour_copy.pop_duration_delta(instance, [old_pos]).total_seconds()
            tour_copy.pop_and_update(instance, [old_pos])
            # check new insertion positions
            for new_pos in range(1, len(tour_copy) - 1):
                if new_pos == old_pos:
                    continue
                insertion_duration_delta = tour_copy.insert_duration_delta(instance, [new_pos],
                                                                           [vertex]).total_seconds()
                delta += insertion_duration_delta
                move = (delta, tour_copy, old_pos, vertex, new_pos)
                # yield the move (with the original tour, not the copy) if it's feasible
                if self.feasibility_check(instance, move):
                    move = (delta, tour, old_pos, vertex, new_pos)
                    yield move
                # restore delta before checking the next insertion position
                delta -= insertion_duration_delta
            # repair the copy before checking the next request for repositioning
            tour_copy.insert_and_update(instance, [old_pos], [vertex])
        pass


class VRPTWTwoOptDist(IntraTourNeighborhood):
    """
    The classic 2-opt neighborhood by Croes (1958)
    """

    def feasible_move_generator_for_tour(self, instance: it.CAHDInstance, tour: tr.Tour):
        for i in range(0, len(tour) - 3):
            for j in range(i + 2, len(tour) - 1):
                delta = 0.0
                delta -= instance.travel_distance(
                    tour.routing_sequence[i:j + 1], tour.routing_sequence[i + 1:j + 2]
                )
                delta += instance.travel_distance(
                    [tour.routing_sequence[i], *tour.routing_sequence[j:i:-1]],
                    [tour.routing_sequence[j], *tour.routing_sequence[j - 1:i:-1], tour.routing_sequence[j + 1]]
                )
                move: vrptw_2opt_move = (delta, tour, i, j)
                if self.feasibility_check(instance, move):
                    yield move

    def feasibility_check(self, instance: it.CAHDInstance, move: vrptw_2opt_move):
        delta, tour, i, j = move

        pop_indices = list(range(i + 1, j + 1))
        popped = tour.pop_and_update(instance, pop_indices)
        feasibility = tour.insertion_feasibility_check(instance, range(i + 1, j + 1), list(reversed(popped)))
        tour.insert_and_update(instance, pop_indices, popped)
        return feasibility

    def execute_move(self, instance: it.CAHDInstance, move: vrptw_2opt_move):
        delta, tour, i, j = move
        popped = tour.pop_and_update(instance, list(range(i + 1, j + 1)))
        tour.insert_and_update(instance, range(i + 1, j + 1), list(reversed(popped)))
        pass


class VRPTWTwoOptDur(IntraTourNeighborhood):
    """
    The classic 2-opt neighborhood by Croes (1958)
    """

    def feasible_move_generator_for_tour(self, instance: it.CAHDInstance, tour: tr.Tour):
        for i in range(0, len(tour) - 3):
            for j in range(i + 2, len(tour) - 1):
                delta = 0.0
                delta -= instance.travel_duration(
                    tour.routing_sequence[i:j + 1], tour.routing_sequence[i + 1:j + 2]
                ).total_seconds()
                delta += instance.travel_duration(
                    [tour.routing_sequence[i], *tour.routing_sequence[j:i:-1]],
                    [tour.routing_sequence[j], *tour.routing_sequence[j - 1:i:-1], tour.routing_sequence[j + 1]]
                ).total_seconds()
                move: vrptw_2opt_move = (delta, tour, i, j)
                # logger.debug(msg=f'testing 2-opt move: {move}')
                if self.feasibility_check(instance, move):
                    yield move

    def feasibility_check(self, instance: it.CAHDInstance, move: vrptw_2opt_move):
        delta, tour, i, j = move

        pop_indices = list(range(i + 1, j + 1))
        popped = tour.pop_and_update(instance, pop_indices)
        feasibility = tour.insertion_feasibility_check(instance, range(i + 1, j + 1), list(reversed(popped)))
        tour.insert_and_update(instance, pop_indices, popped)
        return feasibility

    def execute_move(self, instance: it.CAHDInstance, move: vrptw_2opt_move):
        delta, tour, i, j = move
        popped = tour.pop_and_update(instance, list(range(i + 1, j + 1)))
        tour.insert_and_update(instance, range(i + 1, j + 1), list(reversed(popped)))
        pass


class VRPTWTwoOptDurMax4(VRPTWTwoOptDur):
    def feasible_move_generator_for_tour(self, instance: it.CAHDInstance, tour: tr.Tour):
        for i in range(0, len(tour) - 3):
            for j in range(i + 2, min(i + 6, len(tour) - 1)):
                delta = 0.0
                delta -= instance.travel_duration(
                    tour.routing_sequence[i:j + 1], tour.routing_sequence[i + 1:j + 2]
                ).total_seconds()
                delta += instance.travel_duration(
                    [tour.routing_sequence[i], *tour.routing_sequence[j:i:-1]],
                    [tour.routing_sequence[j], *tour.routing_sequence[j - 1:i:-1], tour.routing_sequence[j + 1]]
                ).total_seconds()
                move: vrptw_2opt_move = (delta, tour, i, j)
                # logger.debug(msg=f'testing 2-opt move: {move}')
                if self.feasibility_check(instance, move):
                    yield move


# =====================================================================================================================
# INTER-TOUR NEIGHBORHOOD
# =====================================================================================================================
class InterTourNeighborhood(Neighborhood, ABC):
    @abstractmethod
    def feasible_move_generator_for_carrier(self, instance: it.CAHDInstance, carrier: slt.AHDSolution):
        """
        :return: tuple containing all necessary information to see whether to accept a move and the information to
        execute a move. The first element must be the delta in travel distance.
        """
        pass

    @abstractmethod
    def feasibility_check(self, instance: it.CAHDInstance, move):
        pass


class VRPTWRelocateDist(InterTourNeighborhood):
    """
    take a delivery request and see whether inserting it intro another tour is cheaper
    """

    def feasible_move_generator_for_carrier(self, instance: it.CAHDInstance, carrier: slt.AHDSolution):
        for old_tour in carrier.tours:
            for old_pos, vertex in enumerate(old_tour.routing_sequence[1:-1], start=1):
                for new_tour in carrier.tours:
                    if new_tour is old_tour:
                        continue
                    for new_pos in range(1, len(new_tour) - 1):
                        delta = 0.0
                        delta += old_tour.pop_distance_delta(instance, [old_pos])
                        delta += new_tour.insert_distance_delta(instance, [new_pos], [vertex])
                        move: vrptw_relocate_move = (delta, carrier, old_tour, old_pos, new_tour, new_pos)
                        if self.feasibility_check(instance, move):
                            yield move
        pass

    def feasibility_check(self, instance: it.CAHDInstance, move: vrptw_relocate_move):
        delta, carrier, old_tour, old_pos, new_tour, new_pos = move
        vertex = old_tour.routing_sequence[old_pos]
        return new_tour.insertion_feasibility_check(instance, [new_pos], [vertex])

    def execute_move(self, instance: it.CAHDInstance, move: vrptw_relocate_move):
        delta, carrier, old_tour, old_pos, new_tour, new_pos = move
        vertex = old_tour.pop_and_update(instance, [old_pos])[0]
        request = instance.request_from_vertex(vertex)
        old_tour.requests.remove(request)

        carrier.drop_empty_tours()
        new_tour.insert_and_update(instance, [new_pos], [vertex])
        new_tour.requests.add(request)
        pass


class VRPTWRelocateDur(InterTourNeighborhood):
    """
    take a delivery request and see whether inserting it intro another tour is cheaper
    """

    def feasible_move_generator_for_carrier(self, instance: it.CAHDInstance, carrier: slt.AHDSolution):
        for old_tour in carrier.tours:
            for old_pos, vertex in enumerate(old_tour.routing_sequence[1:-1], start=1):
                for new_tour in carrier.tours:
                    if new_tour is old_tour:
                        continue
                    for new_pos in range(1, len(new_tour) - 1):
                        delta = 0.0
                        delta += old_tour.pop_duration_delta(instance, [old_pos]).total_seconds()
                        delta += new_tour.insert_duration_delta(instance, [new_pos], [vertex]).total_seconds()
                        move: vrptw_relocate_move = (delta, carrier, old_tour, old_pos, new_tour, new_pos)
                        if self.feasibility_check(instance, move):
                            yield move
        pass

    def feasibility_check(self, instance: it.CAHDInstance, move: vrptw_relocate_move):
        delta, carrier, old_tour, old_pos, new_tour, new_pos = move
        vertex = old_tour.routing_sequence[old_pos]
        return new_tour.insertion_feasibility_check(instance, [new_pos], [vertex])

    def execute_move(self, instance: it.CAHDInstance, move: vrptw_relocate_move):
        delta, carrier, old_tour, old_pos, new_tour, new_pos = move
        vertex = old_tour.pop_and_update(instance, [old_pos])[0]
        request = instance.request_from_vertex(vertex)
        old_tour.requests.remove(request)

        carrier.drop_empty_tours()
        new_tour.insert_and_update(instance, [new_pos], [vertex])
        new_tour.requests.add(request)
        pass
