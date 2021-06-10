import logging
from abc import ABC, abstractmethod
from typing import final

from src.cr_ahd.core_module import instance as it, solution as slt, tour as tr
from src.cr_ahd.routing_module import tour_construction as cns
from src.cr_ahd.utility_module import utils as ut

logger = logging.getLogger(__name__)


# TODO implement a best-improvement vs first-improvement mechanism on the parent-class level. e.g. as a self.best:bool
class LocalSearchBehavior(ABC):

    def local_search(self, instance: it.PDPInstance, solution: slt.CAHDSolution,
                     best_improvement: bool = False):
        for carrier in range(instance.num_carriers):
            self.improve_carrier_solution(instance, solution, carrier, best_improvement)
        pass

    @abstractmethod
    def improve_carrier_solution(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int,
                                 best_improvement: bool):
        pass


'''
class PDPGradientDescent(TourImprovementBehavior):
    """
    As a the most basic "metaheuristic". Whether or not solutions are accepted should then happen on this level. Here:
    always accept only improving solutions
    """

    def local_search(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        pass

    def improve_carrier_solution(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int):
        pass

    def acceptance_criterion(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, delta,
                             history):
        pass
'''


# =====================================================================================================================
# INTRA-TOUR LOCAL SEARCH
# =====================================================================================================================


class PDPIntraTourLocalSearch(LocalSearchBehavior, ABC):

    @final
    def improve_carrier_solution(self,
                                 instance: it.PDPInstance,
                                 solution: slt.CAHDSolution,
                                 carrier: int,
                                 best_improvement: bool):
        for tour in range(solution.carriers[carrier].num_tours()):
            self.improve_tour(instance, solution, carrier, tour, best_improvement)
        pass

    def improve_tour(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, tour: int,
                     best_improvement: bool):
        """
        :param best_improvement: whether to follow a first improvement strategy (False, default) or a best improvement
        (True) strategy

        finds and executes local search moves as long as they yield an improvement.


        """
        improved = True
        while improved:
            improved = False
            move = self.find_feasible_move(instance, solution, carrier, tour, best_improvement)

            # the first element of a move tuple is always its delta
            if self.acceptance_criterion(move[0]):
                logger.debug(f'Intra Tour Local Search move found:')
                self.execute_move(instance, solution, carrier, tour, move)
                improved = True

    @abstractmethod
    def find_feasible_move(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, tour: int,
                           best_improvement: bool):
        """
        :param best_improvement: False (default) if the FIRST improving move shall be executed; True if the BEST
        improving move shall be executed

        :return: tuple containing all necessary information to see whether to accept a move and the information to
        execute a move. The first element must be the delta.
        """
        pass

    def acceptance_criterion(self, delta):  # acceptance criteria could even be their own class
        if delta < 0:
            return True
        else:
            return False

    @abstractmethod
    def execute_move(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, tour: int, move):
        pass

    @abstractmethod
    def feasibility_check(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, tour: int, move):
        pass


class PDPMove(PDPIntraTourLocalSearch, ABC):
    """
    Take a PD pair and see whether inserting it in a different location of the SAME route improves the solution
    """

    def find_feasible_move(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, tour: int,
                           best_improvement: bool):
        tour_ = solution.carriers[carrier].tours[tour]

        # initialize the best known solution
        best_delta = 0
        best_old_pickup_pos = None
        best_old_delivery_pos = None
        best_new_pickup_pos = None
        best_new_delivery_pos = None
        best_move = (best_delta, best_old_pickup_pos, best_old_delivery_pos, best_new_pickup_pos, best_new_delivery_pos)

        # test all requests
        for old_pickup_pos in range(1, len(tour_) - 2):
            vertex = tour_.routing_sequence[old_pickup_pos]

            # skip if its a delivery vertex
            if instance.vertex_type(vertex) == "delivery":
                continue

            pickup, delivery = instance.pickup_delivery_pair(instance.request_from_vertex(vertex))
            old_delivery_pos = tour_.routing_sequence.index(delivery)

            # check all possible new insertions for pickup and delivery vertex of the request
            for new_pickup_pos in range(1, len(tour_) - 2):
                for new_delivery_pos in range(new_pickup_pos + 1, len(tour_) - 1):
                    if new_pickup_pos == old_pickup_pos and new_delivery_pos == old_delivery_pos:
                        continue

                    # create a temporary copy to loop over
                    tmp_routing_sequence = list(tour_.routing_sequence)

                    delta = 0

                    # savings of removing request vertices from their current positions
                    for old_pos in (old_delivery_pos, old_pickup_pos):
                        vertex = tmp_routing_sequence[old_pos]
                        predecessor = tmp_routing_sequence[old_pos - 1]
                        successor = tmp_routing_sequence[old_pos + 1]
                        delta -= instance.distance([predecessor, vertex], [vertex, successor])
                        delta += instance.distance([predecessor], [successor])
                        tmp_routing_sequence.pop(old_pos)

                    # cost for inserting request vertices in the new positions
                    for vertex, new_pos in zip((pickup, delivery), (new_pickup_pos, new_delivery_pos)):
                        predecessor = tmp_routing_sequence[new_pos - 1]
                        successor = tmp_routing_sequence[new_pos]
                        delta += instance.distance([predecessor, vertex], [vertex, successor])
                        delta -= instance.distance([predecessor], [successor])
                        tmp_routing_sequence.insert(new_pos, vertex)

                    # is the current move an improvement?
                    if delta < best_delta:
                        move = (delta, old_pickup_pos, old_delivery_pos, new_pickup_pos, new_delivery_pos)

                        # is the improving move feasible?
                        if self.feasibility_check(instance, solution, carrier, tour, move):

                            # best improvement: update the best known solution and continue
                            if best_improvement:
                                best_delta = delta
                                best_move = move

                            # first improvement: return the move
                            else:
                                return move
        return best_move

    def feasibility_check(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, tour: int, move):
        delta, old_pickup_pos, old_delivery_pos, new_pickup_pos, new_delivery_pos = move
        assert delta < 0, f'Move is non-improving'
        tour_ = solution.carriers[carrier].tours[tour]

        # create a temporary routing sequence to loop over the one that contains the new vertices
        tmp_routing_sequence = list(tour_.routing_sequence)
        delivery = tmp_routing_sequence.pop(old_delivery_pos)
        pickup = tmp_routing_sequence.pop(old_pickup_pos)
        tmp_routing_sequence.insert(new_pickup_pos, pickup)
        tmp_routing_sequence.insert(new_delivery_pos, delivery)

        # find the index from which constraints must be checked
        check_from_index = min(old_pickup_pos, old_delivery_pos, new_pickup_pos, new_delivery_pos)

        return tr.feasibility_check(instance, solution, carrier, tour, tmp_routing_sequence, check_from_index)

    def execute_move(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, tour: int, move):
        delta, old_pickup_pos, old_delivery_pos, new_pickup_pos, new_delivery_pos = move
        tour_ = solution.carriers[carrier].tours[tour]

        # todo: better pop_no_update? saves time!
        pickup, delivery = tour_.pop_and_update(instance, solution, [old_pickup_pos, old_delivery_pos])
        tour_.insert_and_update(instance, solution, [new_pickup_pos, new_delivery_pos], [pickup, delivery])

        pass


class PDPTwoOpt(PDPIntraTourLocalSearch):
    def find_feasible_move(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, tour: int,
                           best_improvement: bool):
        t = solution.carriers[carrier].tours[tour]

        # initialize the best known solution
        best_pos_i = None
        best_pos_j = None
        best_delta = 0
        best_move = (best_delta, best_pos_i, best_pos_j)

        # iterate over all moves
        for i in range(0, len(t) - 3):
            for j in range(i + 2, len(t) - 1):

                delta = 0

                # savings of removing the edges (i, i+1) and (j, j+1)
                delta -= instance.distance([t.routing_sequence[i], t.routing_sequence[j]],
                                           [t.routing_sequence[i + 1], t.routing_sequence[j + 1]])

                # cost of adding the edges (i, j) and (i+1, j+1)
                delta += instance.distance([t.routing_sequence[i], t.routing_sequence[i + 1]],
                                           [t.routing_sequence[j], t.routing_sequence[j + 1]])

                # is the current move an improvement?
                if delta < best_delta:
                    move = (delta, i, j)

                    # is the improving move feasible?
                    if self.feasibility_check(instance, solution, carrier, tour, move):

                        # best improvement: update the best known solution and continue
                        if best_improvement:
                            best_delta = delta
                            best_move = move

                        # first improvement: return the move
                        else:
                            return move

        return best_move

    def feasibility_check(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, tour: int, move):
        delta, i, j = move
        assert delta < 0, f'Move is non-improving'
        tour_ = solution.carriers[carrier].tours[tour]

        # old implementation, may be a bit faster but for now I'll stick to the naive approach
        # also I was not able to follow this old code today and thus, there might be critical bugs
        """
        # check the section-to-be-reverted in reverse order (
        tmp_routing_section_reversed = [i, *range(j, i, -1), *range(j + 1, len(tour_))]
        for rev_idx in range(len(tmp_routing_section_reversed) - 1):
            vertex = tour_.routing_sequence[tmp_routing_section_reversed[rev_idx + 1]]
            predecessor = tour_.routing_sequence[tmp_routing_section_reversed[rev_idx]]

            # time window check
            if solution.tw_close[vertex] < solution.tw_open[predecessor]:
                return False
            else:
                dist = instance.distance([predecessor], [vertex])
                arrival = arrival + ut.travel_time(dist)
                arrival = max(solution.tw_open[predecessor], arrival)
                if arrival > solution.tw_close[vertex]:
                    return False

            # precedence must only be checked if j is a delivery vertex
            if instance.vertex_type(vertex) == "delivery":
                pickup, _ = instance.pickup_delivery_pair(instance.request_from_vertex(vertex))
                for l in range(i + 1, j):
                    if tour_.routing_sequence[l] == pickup:
                        return False

            # load check
            load += instance.load[vertex]
            if load > instance.vehicles_max_load:
                return False

        # if no feasibility check failed
        return True
        """

        # create a temporary routing sequence to loop over the one that contains the reversed section
        tmp_routing_sequence = list(tour_.routing_sequence)
        tmp_routing_sequence = tmp_routing_sequence[:i + 1] + tmp_routing_sequence[j:i:-1] + tmp_routing_sequence[
                                                                                             j + 1:]

        # find the index from which constraints must be checked
        check_from_index = i

        return tr.feasibility_check(instance, solution, carrier, tour, tmp_routing_sequence, check_from_index)

    def execute_move(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, tour: int, move):
        _, i, j = move
        solution.carriers[carrier].tours[tour].reverse_section(instance, solution, i, j)


# =====================================================================================================================
# INTER-TOUR LOCAL SEARCH
# =====================================================================================================================


class PDPInterTourLocalSearch(LocalSearchBehavior):
    @final
    def improve_carrier_solution(self, instance: it.PDPInstance, solution: slt.CAHDSolution,
                                 carrier: int, best_improvement: bool):
        improved = True
        while improved:
            improved = False
            move = self.find_feasible_move(instance, solution, carrier, best_improvement)

            # the first element of a move tuple is always its delta
            if self.acceptance_criterion(move[0]):
                logger.debug(f'Inter Tour Local Search move found:')
                self.execute_move(instance, solution, carrier, move)
                improved = True

        pass

    @abstractmethod
    def find_feasible_move(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int,
                           best_improvement: bool):
        """
        :param best_improvement: False (default) if the FIRST improving move shall be executed; True if the BEST
        improving move shall be executed

        :return: tuple containing all necessary information to see whether to accept a move and the information to
        execute a move. The first element must be the delta.
        """
        pass

    def acceptance_criterion(self, delta):  # acceptance criteria could even be their own class
        if delta < 0:
            return True
        else:
            return False

    @abstractmethod
    def execute_move(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, move):
        pass

    @abstractmethod
    def feasibility_check(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, move):
        pass


class PDPRelocate(PDPInterTourLocalSearch):
    """
    Take one PD request at a time and see whether inserting it into another tour is cheaper.
    """

    def find_feasible_move(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int,
                           best_improvement: bool):
        carrier_ = solution.carriers[carrier]

        # initialize the best known solution
        best_delta = 0
        best_old_tour = None
        best_old_pickup_pos = None
        best_old_delivery_pos = None
        best_new_tour = None
        best_new_pickup_pos = None
        best_new_delivery_pos = None
        best_move = (
            best_delta, best_old_tour, best_old_pickup_pos, best_old_delivery_pos, best_new_tour, best_new_pickup_pos,
            best_new_delivery_pos)

        for old_tour in range(carrier_.num_tours()):

            # check all requests of the old tour
            # a pickup can be at most at index n-2 (n is depot, n-1 must be delivery)
            for old_pickup_pos in range(1, len(carrier_.tours[old_tour]) - 2):
                vertex = carrier_.tours[old_tour].routing_sequence[old_pickup_pos]

                # skip if its a delivery vertex
                if instance.vertex_type(vertex) == "delivery":
                    continue  # skip if its a delivery vertex

                pickup, delivery = instance.pickup_delivery_pair(instance.request_from_vertex(vertex))
                old_delivery_pos = carrier_.tours[old_tour].routing_sequence.index(delivery)

                pickup_predecessor = carrier_.tours[old_tour].routing_sequence[old_pickup_pos - 1]
                pickup_successor = carrier_.tours[old_tour].routing_sequence[old_pickup_pos + 1]
                delivery_predecessor = carrier_.tours[old_tour].routing_sequence[old_delivery_pos - 1]
                delivery_successor = carrier_.tours[old_tour].routing_sequence[old_delivery_pos + 1]

                delta = 0

                # differentiate between cases were
                # (a) pickup and delivery are in immediate succession ...
                if delivery_predecessor == pickup:

                    # savings of removing request from current tour
                    delta -= instance.distance([pickup_predecessor, pickup, delivery],
                                               [pickup, delivery, delivery_successor])

                    # cost of reconnecting the lose ends
                    delta += instance.distance([pickup_predecessor], [delivery_successor])

                # ... or (b) there are other vertices between pickup and delivery
                else:

                    # savings of removing request from current tour
                    delta -= instance.distance([pickup_predecessor, pickup, delivery_predecessor, delivery],
                                               [pickup, pickup_successor, delivery, delivery_successor])

                    # cost of reconnecting the lose ends
                    delta += instance.distance([pickup_predecessor, delivery_predecessor],
                                               [pickup_successor, delivery_successor])

                # cost for inserting request into another tour
                # check all new tours
                for new_tour in range(carrier_.num_tours()):
                    new_tour_ = carrier_.tours[new_tour]

                    # skip the current tour, there is the PDPMove local search for that option
                    if new_tour == old_tour:
                        continue

                    # check all possible new insertions for pickup and delivery vertex of the request
                    for new_pickup_pos in range(1, len(new_tour_) - 1):
                        for new_delivery_pos in range(new_pickup_pos + 1, len(new_tour_)):
                            delta += new_tour_.insertion_distance_delta(
                                instance, [new_pickup_pos, new_delivery_pos], [pickup, delivery])

                            # is the current move an improvement?
                            if delta < best_delta:
                                move = (delta, old_tour, old_pickup_pos, old_delivery_pos, new_tour, new_pickup_pos,
                                        new_delivery_pos)

                                # is the improving move feasible?
                                if self.feasibility_check(instance, solution, carrier, move):

                                    # best improvement: update the best known solution and continue
                                    if best_improvement:
                                        best_delta = delta
                                        best_move = move

                                    # first improvement: return the move
                                    else:
                                        return move

        return best_move

    def feasibility_check(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, move):
        delta, old_tour, old_pickup_pos, old_delivery_pos, new_tour, new_pickup_pos, new_delivery_pos = move
        assert delta < 0, f'Move is non-improving'
        old_tour_ = solution.carriers[carrier].tours[old_tour]
        new_tour_ = solution.carriers[carrier].tours[new_tour]

        # create a temporary copy of the routing sequence to check feasibility
        tmp_routing_sequence = list(new_tour_.routing_sequence)
        pickup = old_tour_.routing_sequence[old_pickup_pos]
        delivery = old_tour_.routing_sequence[old_delivery_pos]
        tmp_routing_sequence.insert(new_pickup_pos, pickup)
        tmp_routing_sequence.insert(new_delivery_pos, delivery)

        # find the index from which constraints must be checked
        check_from_index = new_pickup_pos

        return tr.feasibility_check(instance, solution, carrier, new_tour, tmp_routing_sequence, check_from_index)


    def execute_move(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, move):
        delta, old_tour, old_pickup_pos, old_delivery_pos, new_tour, new_pickup_pos, new_delivery_pos = move
        carrier_ = solution.carriers[carrier]

        # todo: better pop_no_update? saves time!
        carrier_.tours[old_tour].pop_and_update(instance, solution, [old_pickup_pos, old_delivery_pos])
        carrier_.tours[new_tour].insert_and_update(instance, solution, [new_pickup_pos, new_delivery_pos],
                                                   [(carrier_.tours[old_tour].routing_sequence[old_pickup_pos]),
                                                    (carrier_.tours[old_tour].routing_sequence[old_delivery_pos])])
        pass

# class ThreeOpt(TourImprovementBehavior):
#     pass

# class LinKernighan(TourImprovementBehavior):
#     pass

# class Swap(TourImprovementBehavior):
#     pass
