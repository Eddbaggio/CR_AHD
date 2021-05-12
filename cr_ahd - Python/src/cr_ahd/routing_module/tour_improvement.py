import logging
from abc import ABC, abstractmethod
from typing import final

from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import tour_construction as cns
from src.cr_ahd.utility_module import utils as ut

logger = logging.getLogger(__name__)


# TODO implement a best-improvement vs first-improvement mechanism on the parent-class level. e.g. as a self.best:bool
class TourImprovementBehavior(ABC):

    @abstractmethod
    def improve_global_solution(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        pass

    @abstractmethod
    def improve_carrier_solution(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int):
        pass


'''
class PDPGradientDescent(TourImprovementBehavior):
    """
    As a the most basic "metaheuristic". Whether or not solutions are accepted should then happen on this level. Here:
    always accept only improving solutions
    """

    def improve_global_solution(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
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


class PDPIntraTourLocalSearch(TourImprovementBehavior, ABC):
    @final
    def improve_global_solution(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        for carrier in range(instance.num_carriers):
            self.improve_carrier_solution(instance, solution, carrier)
        pass

    @final
    def improve_carrier_solution(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int):
        for tour in range(solution.carriers[carrier].num_tours()):
            improved = True
            while improved:
                improved = False
                move = self.find_move(instance, solution, carrier, tour)
                # best_pos_i, best_pos_j, best_delta = self.improve_tour(instance, solution, carrier, tour)
                if self.acceptance_criterion(move[0]):
                    logger.debug(f'Intra Tour Local Search move found:')
                    self.execute_move(instance, solution, carrier, tour, move)
                    improved = True
        pass

    @abstractmethod
    def feasibility_check(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, tour: int, move):
        pass

    def acceptance_criterion(self, delta):  # acceptance criteria could even be their own class
        if delta < 0:
            return True
        else:
            return False

    @abstractmethod
    def execute_move(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, tour: int, move):
        pass

    @abstractmethod
    def find_move(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, tour: int):
        """
        :return: tuple containing all necessary information to see whether to accept a move and the information to
        execute a move. The first element must be the delta
        """
        pass


class PDPTwoOptBest(PDPIntraTourLocalSearch):
    def find_move(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, tour: int):
        t = solution.carriers[carrier].tours[tour]
        best_pos_i = None
        best_pos_j = None
        best_delta = 0  # TODO maybe better to initialize with inf?
        for i in range(0, len(t) - 3):
            for j in range(i + 2, len(t) - 1):
                delta = instance.distance([t.routing_sequence[i], t.routing_sequence[i + 1]],
                                          [t.routing_sequence[j], t.routing_sequence[j + 1]]) - \
                        instance.distance([t.routing_sequence[i], t.routing_sequence[j]],
                                          [t.routing_sequence[i + 1], t.routing_sequence[j + 1]])
                if delta < best_delta:
                    if self.feasibility_check(instance, solution, carrier, tour, (i, j)):
                        best_pos_i = i
                        best_pos_j = j
                        best_delta = delta
        return best_delta, best_pos_i, best_pos_j

    def feasibility_check(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, tour: int, move):
        i, j = move
        tour_ = solution.carriers[carrier].tours[tour]
        arrival = tour_.arrival_schedule[i]
        load = tour_.load_sequence[i]

        # check the section-to-be-reverted in reverse order
        reversed_seq_indices = [i, *range(j, i, -1), *range(j + 1, len(tour_))]
        for rev_idx in range(len(reversed_seq_indices) - 1):
            vertex = tour_.routing_sequence[reversed_seq_indices[rev_idx + 1]]
            predecessor = tour_.routing_sequence[reversed_seq_indices[rev_idx]]

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

    def execute_move(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, tour: int, move):
        _, i, j = move
        solution.carriers[carrier].tours[tour].reverse_section(instance, solution, i, j)


'''
class PDPTwoOptFirstImprovement(PDPIntraTourLocalSearch):
    def improve_tour(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, tour: int):
        t = solution.carrier_solutions[carrier].tours[tour]
        for i in range(0, len(t) - 3):
            for j in range(i + 2, len(t) - 1):
                delta = instance.distance([i, i + 1], [j, j + 1]) - instance.distance([i, j], [i + 1, j + 1])
                if delta < 0:
                    return i, j, delta
'''


class PDPMoveBest(PDPIntraTourLocalSearch, ABC):
    """
    Take a PD pair and see whether inserting it in a different location of the SAME route improves the solution
    """

    def find_move(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, tour: int):
        tour_ = solution.carriers[carrier].tours[tour]

        best_delta = 0
        best_old_pickup_pos = None
        best_old_delivery_pos = None
        best_new_pickup_pos = None
        best_new_delivery_pos = None

        best_move = (best_delta, best_old_pickup_pos, best_old_delivery_pos, best_new_pickup_pos, best_new_delivery_pos)

        for old_pickup_pos in range(1, len(tour_) - 2):

            vertex = tour_.routing_sequence[old_pickup_pos]
            if instance.vertex_type(vertex) == "delivery":
                continue  # skip if its a delivery vertex
            pickup, delivery = instance.pickup_delivery_pair(instance.request_from_vertex(vertex))
            old_delivery_pos = tour_.routing_sequence.index(delivery)

            for new_pickup_pos in range(1, len(tour_) - 2):
                for new_delivery_pos in range(new_pickup_pos + 1, len(tour_) - 1):
                    if new_pickup_pos == old_pickup_pos and new_delivery_pos == old_delivery_pos:
                        continue
                    tmp_routing_sequence = list(tour_.routing_sequence)

                    # savings of removing request vertices from their current positions
                    delta = 0
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

                    if delta < best_delta:
                        move = (delta, old_pickup_pos, old_delivery_pos, new_pickup_pos, new_delivery_pos)
                        if self.feasibility_check(instance, solution, carrier, tour, move):
                            best_delta = delta
                            best_move = move
        return best_move

    '''
    def find_move(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, tour: int):
        cs = solution.carrier_solutions[carrier]
        rs = cs.tours[tour].routing_sequence

        best_delta = 0
        best_old_pickup_pos = None
        best_old_delivery_pos = None
        best_new_pickup_pos = None
        best_new_delivery_pos = None

        best_move = (best_delta, best_old_pickup_pos, best_old_delivery_pos, best_new_pickup_pos, best_new_delivery_pos)

        for old_pickup_pos in range(1, len(cs.tours[tour]) - 2):

            # savings of removing request vertices from their current positions
            vertex = rs[old_pickup_pos]
            if instance.vertex_type(vertex) == "delivery":
                continue  # skip if its a delivery vertex
            pickup, delivery = instance.pickup_delivery_pair(instance.request_from_vertex(vertex))
            old_delivery_pos = rs.index(delivery)

            old_pickup_predecessor = rs[old_pickup_pos - 1]
            old_delivery_successor = rs[old_delivery_pos + 1]

            # there is no vertex between pickup and delivery
            if old_delivery_pos == old_pickup_pos + 1:
                origins_save = [old_pickup_predecessor, pickup, delivery]
                destinations_save = [*origins_save[1:], old_delivery_successor]
                origins_reconnect = [old_pickup_predecessor]
                destinations_reconnect = [old_delivery_successor]

            # there is exactly one vertex between pickup and delivery
            elif old_delivery_pos == old_pickup_pos + 2:
                origins_save = [old_pickup_predecessor, pickup, rs[old_pickup_pos + 2], delivery]
                destinations_save = [*origins_save[1:], old_delivery_successor]
                origins_reconnect = [old_pickup_predecessor, rs[old_pickup_pos + 2]]
                destinations_reconnect = [rs[old_pickup_pos + 2], old_delivery_successor]

            # there is more than one vertex between pickup and delivery
            else:
                origins_save = [old_pickup_predecessor, pickup, rs[old_delivery_pos - 1], delivery]
                destinations_save = [pickup, rs[old_pickup_pos + 2], delivery, old_delivery_successor]
                origins_reconnect = [old_pickup_predecessor, rs[old_delivery_pos - 1]]
                destinations_reconnect = [rs[old_pickup_pos + 2], old_delivery_successor]

            # savings by removing and reconnecting
            savings = -instance.distance(origins_save, destinations_save)
            savings += instance.distance(origins_reconnect, destinations_reconnect)

            # cost for inserting request vertices in the new positions
            for new_pickup_pos in range(1, len(cs.tours[tour]) - 2):
                for new_delivery_pos in range(new_pickup_pos + 1, len(cs.tours[tour]) - 1):
                    if new_pickup_pos == old_pickup_pos and new_delivery_pos == old_delivery_pos:
                        continue
                    else:
                        new_pickup_predecessor = rs[new_pickup_pos - 1]
                        new_delivery_successor = rs[new_delivery_pos + 1]
                        # there is no vertex between pickup and delivery
                        if new_delivery_pos == new_pickup_pos + 1:
                            origins_save = [new_pickup_predecessor]
                            destinations_save = [new_delivery_successor]
                            origins_insert = [new_pickup_predecessor, pickup, delivery]
                            destinations_insert = [pickup, delivery, new_delivery_successor]

                        # there is exactly one vertex between pickup and delivery
                        elif new_delivery_pos == new_pickup_pos + 2:
                            origins_save = [new_pickup_predecessor, rs[new_pickup_pos + 2]]
                            destinations_save = [rs[new_pickup_pos + 2], new_delivery_successor]
                            origins_insert = [new_pickup_predecessor, pickup, rs[new_pickup_pos + 2], delivery]
                            destinations_insert = [pickup, rs[new_pickup_pos + 2], delivery, new_delivery_successor]

                        # there is more than one vertex between pickup and delivery
                        else:
                            origins_save = [new_pickup_predecessor, rs[new_delivery_pos]]
                            destinations_save = [rs[new_pickup_pos + 2], new_delivery_successor]
                            origins_insert = [new_pickup_predecessor, pickup, rs[new_delivery_pos], delivery]
                            destinations_insert = [pickup, rs[new_pickup_pos + 2], delivery, new_delivery_successor]

                    # update the delta with the insertion costs
                    delta = savings - instance.distance(origins_save, destinations_save) + instance.distance(
                        origins_insert, destinations_insert)

                    # is the current move better than the best known move?
                    move = (delta, old_pickup_pos, old_delivery_pos, new_pickup_pos, new_delivery_pos)
                    if delta < best_delta:
                        if self.feasibility_check(instance, solution, carrier, tour, move):
                            best_delta = delta
                            best_move = move

        return best_move
        '''

    def feasibility_check(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, tour: int, move):
        delta, old_pickup_pos, old_delivery_pos, new_pickup_pos, new_delivery_pos = move
        tour_ = solution.carriers[carrier].tours[tour]
        # create a temporary routing sequence to loop over that contains the new vertices
        tmp_routing_sequence = list(tour_.routing_sequence)
        delivery = tmp_routing_sequence.pop(old_delivery_pos)
        pickup = tmp_routing_sequence.pop(old_pickup_pos)
        tmp_routing_sequence.insert(new_pickup_pos, pickup)
        tmp_routing_sequence.insert(new_delivery_pos, delivery)

        total_travel_distance = sum(tour_.travel_distance_sequence[:old_pickup_pos])
        service_time = tour_.service_schedule[old_pickup_pos - 1]
        load = tour_.load_sequence[old_pickup_pos - 1]

        # iterate over the temporary tour and check all constraints
        for pos in range(old_pickup_pos, len(tmp_routing_sequence)):
            vertex: int = tmp_routing_sequence[pos]
            predecessor_vertex: int = tmp_routing_sequence[pos - 1]

            # check precedence
            if instance.vertex_type(vertex) == "delivery":
                precedence_vertex = vertex - instance.num_requests
                if precedence_vertex not in tmp_routing_sequence[:pos]:
                    return False

            # check max tour distance
            travel_distance = instance.distance([predecessor_vertex], [vertex])
            total_travel_distance += travel_distance
            if total_travel_distance > instance.vehicles_max_travel_distance:
                return False

            # check time windows
            arrival_time = service_time + instance.service_duration[predecessor_vertex] + ut.travel_time(travel_distance)
            if arrival_time > solution.tw_close[vertex]:
                return False
            service_time = max(arrival_time, solution.tw_open[vertex])

            # check max vehicle load
            load += instance.load[vertex]
            if load > instance.vehicles_max_load:
                return False
        return True

    def execute_move(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, tour: int, move):
        delta, old_pickup_pos, old_delivery_pos, new_pickup_pos, new_delivery_pos = move
        tour_ = solution.carriers[carrier].tours[tour]

        pickup, delivery = tour_.pop_and_update(instance, solution, [old_pickup_pos, old_delivery_pos])
        tour_.insert_and_update(instance, solution, [new_pickup_pos, new_delivery_pos], [pickup, delivery])

        pass


# =====================================================================================================================
# INTER-TOUR LOCAL SEARCH
# =====================================================================================================================


class PDPInterTourLocalSearch(TourImprovementBehavior):
    @final
    def improve_global_solution(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        for carrier in range(instance.num_carriers):
            improved = True
            while improved:
                improved = False
                move = self.find_move(instance, solution, carrier)
                if self.acceptance_criterion(move[0]):
                    logger.debug(f'Inter Tour Local Search move found:')
                    self.execute_move(instance, solution, carrier, move)
                    improved = True
        pass

    @final
    def improve_carrier_solution(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int):
        pass

    @abstractmethod
    def feasibility_check(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, move):
        """must be used in the find_move method"""
        pass

    def acceptance_criterion(self, delta):  # acceptance criteria could even be their own class
        if delta < 0:
            return True
        else:
            return False

    @abstractmethod
    def execute_move(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, move):
        pass

    @abstractmethod
    def find_move(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int):
        """
        finds the best/first feasible move

        :return: tuple containing all necessary information to see whether to accept a move and the information to
        execute a move. The first element must be the delta
        """
        pass


class PDPExchangeMoveBest(PDPInterTourLocalSearch):
    """
    Take one PD request at a time and see whether inserting it into another tour is cheaper.
    BEST improvement for each PD request.
    """

    def find_move(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int):
        raise NotImplementedError
        carrier_ = solution.carriers[carrier]

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
                delta = 0

                vertex = carrier_.tours[old_tour].routing_sequence[old_pickup_pos]
                if instance.vertex_type(vertex) == "delivery":
                    continue  # skip if its a delivery vertex
                pickup, delivery = instance.pickup_delivery_pair(instance.request_from_vertex(vertex))
                old_delivery_pos = carrier_.tours[old_tour].routing_sequence.index(delivery)
                pickup_predecessor = carrier_.tours[old_tour].routing_sequence[old_pickup_pos - 1]
                pickup_successor = carrier_.tours[old_tour].routing_sequence[old_pickup_pos + 1]
                delivery_predecessor = carrier_.tours[old_tour].routing_sequence[old_delivery_pos - 1]
                delivery_successor = carrier_.tours[old_tour].routing_sequence[old_delivery_pos + 1]

                # savings of removing request from current tour
                if delivery_predecessor == pickup:
                    delta = delta - instance.distance(
                        [pickup_predecessor, pickup, delivery],
                        [pickup, delivery, delivery_successor]) + instance.distance([pickup_predecessor],
                                                                                    [delivery_successor])
                else:
                    delta = delta - instance.distance(
                        [pickup_predecessor, pickup, delivery_predecessor, delivery],
                        [pickup, pickup_successor, delivery, delivery_successor]) + instance.distance(
                        [pickup_predecessor, delivery_predecessor], [pickup_successor, delivery_successor])

                # cost for inserting request into another tour
                for new_tour in range(carrier_.num_tours()):
                    if new_tour == old_tour:
                        continue
                    for new_pickup_pos in range(1, len(carrier_.tours[new_tour]) - 1):
                        for new_delivery_pos in range(new_pickup_pos + 1, len(carrier_.tours[new_tour])):
                            insertion_cost = carrier_.tours[new_tour].insertion_distance_delta(
                                instance, [new_pickup_pos, new_delivery_pos], [pickup, delivery])
                            delta += insertion_cost

                            # is the current move better than the best known move?
                            move = (delta, old_tour, old_pickup_pos, old_delivery_pos, new_tour, new_pickup_pos,
                                    new_delivery_pos)
                            if delta < best_move[0]:
                                if self.feasibility_check(instance, solution, carrier, move):
                                    best_move = move

        return best_move

    def feasibility_check(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, move):
        # TODO check ALL constraints
        delta, old_tour, old_pickup_pos, old_delivery_pos, new_tour, new_pickup_pos, new_delivery_pos = move
        carrier_ = solution.carriers[carrier]
        return carrier_.tours[new_tour].insertion_feasibility_check(instance, solution,
                                                                    [new_pickup_pos, new_delivery_pos],
                                                                    [(carrier_.tours[old_tour].routing_sequence[
                                                                        old_pickup_pos]),
                                                                     (carrier_.tours[old_tour].routing_sequence[
                                                                         old_delivery_pos])])

    def execute_move(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, move):
        delta, old_tour, old_pickup_pos, old_delivery_pos, new_tour, new_pickup_pos, new_delivery_pos = move
        carrier_ = solution.carriers[carrier]
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
