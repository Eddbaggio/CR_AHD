import logging
import random
import time
from abc import abstractmethod, ABC
from copy import deepcopy
from math import exp, log
from typing import Sequence, List, Union

from src.cr_ahd.core_module import instance as it, solution as slt, tour as tr
from src.cr_ahd.routing_module import neighborhoods as nh, shakes as sh, tour_construction as cns
from src.cr_ahd.utility_module import utils as ut, profiling as pr

logger = logging.getLogger(__name__)

if ut.debugger_is_active():
    TIME_MAX = float(1)  # 0.05 is roughly the time required by the VND procedure to exhaust all neighborhoods
    ITER_MAX = float('inf')  # better to use iter_max in some cases to obtain same results in post-acceptance & bidding
else:
    TIME_MAX = float(1)  # 0.05 is roughly the time required by the VND procedure to exhaust all neighborhoods
    ITER_MAX = float('inf')  # better to use iter_max in some cases to obtain same results in post-acceptance & bidding


class PDPTWMetaHeuristic(ABC):
    def __init__(self, neighborhoods: Sequence[nh.Neighborhood]):
        self.neighborhoods = neighborhoods
        self.improved = False
        self.start_time = None
        self.iter_count = None
        self.parameters = dict()
        self.trajectory = []  # collection of all accepted & executed moves

        self.name = f'{self.__class__.__name__}'

    @abstractmethod
    def execute(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution,
                carrier_ids: List[int] = None) -> slt.CAHDSolution:
        pass

    # @abstractmethod
    # def execute_on_carrier(self, instance: it.MDPDPTWInstance, carrier: slt.AHDSolution):
    #     pass

    @abstractmethod
    def acceptance_criterion(self, instance: it.MDPDPTWInstance, move: tuple):
        return True

    @abstractmethod
    def stopping_criterion(self):
        pass

    def update_trajectory(self, name: str, move: tuple, accepted: bool):
        self.trajectory.append((name, move, accepted))
        pass


class NoMetaheuristic(PDPTWMetaHeuristic):
    """Placeholder for cases in which no improvement is wanted"""

    def stopping_criterion(self):
        pass

    def acceptance_criterion(self, instance: it.MDPDPTWInstance, move: tuple):
        pass

    def execute(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution,
                carrier_ids: List[int] = None) -> slt.CAHDSolution:
        return solution

    def execute_on_carrier(self, instance: it.MDPDPTWInstance, carrier: slt.AHDSolution):
        pass


# class PDPTWIntraTourMetaheuristic(PDPTWMetaHeuristic, ABC):
#     """Metaheuristic that operates only on a single tour, i.e. all its neighborhoods are IntraTourNeighborhood"""
#     def __init__(self, neighborhoods: Sequence[nh.Neighborhood]):
#         for nbh in neighborhoods:
#             assert isinstance(nbh, nh.IntraTourNeighborhood)
#         super().__init__(neighborhoods)
#
#     @abstractmethod
#     def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carriers=None) -> slt.CAHDSolution:
#         pass
#
#     @abstractmethod
#     def execute_on_tour(self, instance:it.PDPInstance, solution:slt.CAHDSolution, tour_=tr.Tour):
#         pass

class LocalSearchFirst(PDPTWMetaHeuristic):
    """
    local search heuristic using the first improvement strategy
    """

    def execute(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution,
                carrier_ids: List[int] = None) -> slt.CAHDSolution:
        assert len(self.neighborhoods) == 1, 'Local Search can use a single neighborhood only!'
        best_solution = deepcopy(solution)
        if carrier_ids is None:
            carrier_ids = [x.id_ for x in best_solution.carriers]

        for carrier_id in carrier_ids:
            carrier = solution.carriers[carrier_id]
            neighborhood = self.neighborhoods[0]
            self.improved = True
            self.start_time = time.time()
            while not self.stopping_criterion():
                self.improved = False
                move_gen = neighborhood.feasible_move_generator_for_carrier(instance, carrier)
                try:
                    move = next(move_gen)  # may be feasible but not improving
                    while not self.acceptance_criterion(instance, move):
                        self.update_trajectory(neighborhood.__class__.__name__, move, False)
                        move = next(move_gen)
                    neighborhood.execute_move(instance, move)
                    self.update_trajectory(neighborhood.__class__.__name__, move, True)
                    self.improved = True
                except StopIteration:
                    break  # exit the while loop (while-condition is false anyway)
        return best_solution

    def acceptance_criterion(self, instance: it.MDPDPTWInstance, move: tuple):
        if move[0] < 0:
            return True
        else:
            return False

    def stopping_criterion(self):
        if self.improved and time.time() - self.start_time < TIME_MAX:
            return False
        else:
            return True


class LocalSearchBest(PDPTWMetaHeuristic):
    """implements a the local search heuristic using the best improvement strategy, i.e. steepest descent"""

    def execute(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution,
                carrier_ids: List[int] = None) -> slt.CAHDSolution:
        assert len(self.neighborhoods) == 1, 'Local Search must have a single neighborhood only!'
        best_solution = deepcopy(solution)
        if carrier_ids is None:
            carrier_ids = [x.id_ for x in best_solution.carriers]

        for carrier_id in carrier_ids:
            carrier = solution.carriers[carrier_id]
            neighborhood = self.neighborhoods[0]
            self.improved = True
            self.start_time = time.time()
            while not self.stopping_criterion():
                self.improved = False
                all_moves = [move for move in neighborhood.feasible_move_generator_for_carrier(instance, carrier)]
                if any(all_moves):
                    best_move = min(all_moves, key=lambda x: x[0])
                    if self.acceptance_criterion(instance, best_move):
                        self.update_trajectory(neighborhood.__class__.__name__, best_move, True)
                        neighborhood.execute_move(instance, best_move)
                        self.improved = True
        return best_solution

    def acceptance_criterion(self, instance: it.MDPDPTWInstance, move: tuple):
        if move[0] < 0:
            return True
        else:
            return False

    def stopping_criterion(self):
        if self.improved and time.time() - self.start_time < TIME_MAX:
            return False
        else:
            return True


class PDPTWSequentialLocalSearch(PDPTWMetaHeuristic):
    """
    Sequentially exhaust each neighborhood in their given order. Only improvements are accepted. First improvement is
    used.
    """

    def execute(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution,
                carrier_ids: List[int] = None) -> slt.CAHDSolution:
        best_solution = deepcopy(solution)
        if carrier_ids is None:
            carrier_ids = [x.id_ for x in best_solution.carriers]

        for carrier_id in carrier_ids:
            carrier = solution.carriers[carrier_id]
            for k in range(len(self.neighborhoods)):
                neighborhood = self.neighborhoods[k]
                move_generator = neighborhood.feasible_move_generator_for_carrier(instance, carrier)
                self.start_time = time.time()
                self.improved = True
                while not self.stopping_criterion():
                    self.improved = False
                    while self.improved is False:
                        try:
                            move = next(move_generator)
                            if self.acceptance_criterion(instance, move):
                                neighborhood.execute_move(instance, move)
                                self.improved = True
                                self.update_trajectory(neighborhood.__class__.__name__, move, True)
                                move_generator = neighborhood.feasible_move_generator_for_carrier(instance, carrier)
                        except StopIteration:
                            # StopIteration occurs if there are no neighbors that can be returned by the move_generator
                            break
        return best_solution

    def acceptance_criterion(self, instance: it.MDPDPTWInstance, move: tuple):
        if move is None:
            return False
        elif move[0] < 0:
            return True
        else:
            return False

    def stopping_criterion(self):
        if self.improved and time.time() - self.start_time < TIME_MAX:
            return False
        else:
            return True


class PDPTWVariableNeighborhoodDescent(PDPTWMetaHeuristic):
    """
    deterministic variant of VNS. multiple neighborhoods are ordered and searched sequentially. In each neighborhood
    that is searched, the *best* found neighbor is used. When stuck in a local optimum, switches to the next
    neighborhood
    """

    def execute(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution,
                carrier_ids: List[int] = None) -> slt.CAHDSolution:
        best_solution = deepcopy(solution)
        if carrier_ids is None:
            carrier_ids = [x.id_ for x in best_solution.carriers]

        for carrier_id in carrier_ids:
            carrier = best_solution.carriers[carrier_id]
            self.parameters['k'] = 0
            self.start_time = time.time()
            while not self.stopping_criterion():
                neighborhood = self.neighborhoods[self.parameters['k']]
                all_moves = [move for move in neighborhood.feasible_move_generator_for_carrier(instance, carrier)]
                if any(all_moves):
                    best_move = min(all_moves, key=lambda x: x[0])
                    if self.acceptance_criterion(instance, best_move):
                        neighborhood.execute_move(instance, best_move)
                        # ut.validate_solution(instance, best_solution)
                        self.update_trajectory(self.parameters['k'], best_move, True)
                        self.parameters['k'] = 0
                    else:
                        self.update_trajectory(self.parameters['k'], best_move, False)
                        self.parameters['k'] += 1
                else:
                    self.parameters['k'] += 1
        return best_solution

    def execute_on_tour(self, instance: it.MDPDPTWInstance, tour: tr.Tour):
        """
        execute the metaheuristic for a given route (in place) using all available intra-tour neighborhoods. useful if
        a tour shall be improved that does not belong to a carrier. E.g. when estimating the tour length of a bundle
        """

        intra_tour_neighborhoods = [nbh for nbh in self.neighborhoods if isinstance(nbh, nh.IntraTourNeighborhood)]
        self.parameters['k'] = 0
        while self.parameters['k'] < len(intra_tour_neighborhoods):
            neighborhood = intra_tour_neighborhoods[self.parameters['k']]
            all_moves = [move for move in neighborhood.feasible_move_generator_for_tour(instance, tour)]
            if any(all_moves):
                best_move = min(all_moves, key=lambda x: x[0])
                if self.acceptance_criterion_tour(best_move):
                    neighborhood.execute_move(instance, best_move)
                    self.update_trajectory(self.parameters['k'], best_move, True)
                    self.parameters['k'] = 0
                else:
                    self.update_trajectory(self.parameters['k'], best_move, False)
                    self.parameters['k'] += 1
            else:
                self.parameters['k'] += 1

    def acceptance_criterion(self, instance: it.MDPDPTWInstance, move: tuple):
        if move[0] < 0:
            return True
        else:
            return False

    def acceptance_criterion_tour(self, move: tuple):
        if move[0] < 0:
            return True
        else:
            return False

    def stopping_criterion(self):
        if self.parameters['k'] < len(self.neighborhoods) and time.time() - self.start_time < TIME_MAX:
            return False
        else:
            return True


class PDPTWReducedVariableNeighborhoodSearch(PDPTWVariableNeighborhoodDescent):
    """
    stochastic variant of VNS. a random neighbor from the current neighborhood is drawn and executed if it improves the
    solution
    """

    def execute(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution,
                carrier_ids: List[int] = None) -> slt.CAHDSolution:
        best_solution = deepcopy(solution)
        if carrier_ids is None:
            carrier_ids = [x.id_ for x in best_solution.carriers]

        for carrier_id in carrier_ids:
            carrier = solution.carriers[carrier_id]
            self.parameters['k'] = 0
            self.start_time = time.time()
            while not self.stopping_criterion():
                neighborhood = self.neighborhoods[self.parameters['k']]
                all_moves = [move for move in neighborhood.feasible_move_generator_for_carrier(instance, carrier)]
                if any(all_moves):
                    random_move = random.choice(all_moves)
                    if self.acceptance_criterion(instance, random_move):
                        neighborhood.execute_move(instance, random_move)
                        # ut.validate_solution(instance, best_solution)
                        self.update_trajectory(self.parameters['k'], random_move, True)
                        self.parameters['k'] = 0
                    else:
                        self.update_trajectory(self.parameters['k'], random_move, False)
                        self.parameters['k'] += 1
                else:
                    self.parameters['k'] += 1
        return best_solution

    def execute_on_tour(self, instance: it.MDPDPTWInstance, tour: tr.Tour):
        raise NotImplementedError()


class PDPTWSimulatedAnnealing(PDPTWMetaHeuristic):
    def __init__(self, neighborhoods: Sequence[nh.Neighborhood]):
        super().__init__(neighborhoods)
        self.parameters['initial_temperature'] = 0
        self.parameters['temperature'] = 0
        self.parameters['cooling_factor'] = 0.85

    def execute(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution,
                carrier_ids: List[int] = None) -> slt.CAHDSolution:
        solution = deepcopy(solution)
        best_solution = deepcopy(solution)
        if carrier_ids is None:
            carrier_ids = [x.id_ for x in best_solution.carriers]

        for carrier_id in carrier_ids:
            carrier = solution.carriers[carrier_id]
            self.parameters['initial_temperature'] = self.compute_start_temperature(carrier)
            self.parameters['temperature'] = self.parameters['initial_temperature']

            self.start_time = time.time()

            i = 0
            while not self.stopping_criterion():
                # update the current temperature
                self.parameters['temperature'] = \
                    self.parameters['initial_temperature'] * self.parameters['cooling_factor'] ** i

                # random neighbor
                neighborhood = random.choice(self.neighborhoods)
                all_moves = list(neighborhood.feasible_move_generator_for_carrier(instance, carrier))
                if any(all_moves):
                    move = random.choice(all_moves)

                    if self.acceptance_criterion(instance, move):
                        neighborhood.execute_move(instance, move)
                        self.update_trajectory(neighborhood.__class__.__name__, move, True)
                        # update the best solution
                        if solution.objective() > best_solution.objective():
                            best_solution = deepcopy(solution)
                else:
                    continue
                i += 1
        return best_solution

    def acceptance_criterion(self, instance: it.MDPDPTWInstance, move: tuple):
        """
        always accept improving moves, accept deteriorating moves with a certain probability.
        """
        try:
            if move is None:
                return False
            # improving move is always accepted
            elif move[0] <= 0:
                return True
            # degrading move is accepted with certain probability
            elif random.random() < exp(-move[0] / self.parameters['temperature']):  # might raise OverflowError
                return True
            else:
                return False
        except OverflowError:
            # overflow caused by double limit if temperature is too low. In that case the random number will most
            # likely not be smaller than the required probability
            return False

    def stopping_criterion(self):
        if time.time() - self.start_time < TIME_MAX and self.parameters['temperature'] > 1:
            return False
        else:
            return True

    @staticmethod
    def compute_start_temperature(carrier: slt.AHDSolution, start_temp_control_param=0.5):
        """compute start temperature according to Ropke,S., & Pisinger,D. (2006). An Adaptive Large Neighborhood
        Search Heuristic for the Pickup and Delivery Problem with Time Windows. Transportation Science, 40(4),
        455–472. https://doi.org/10.1287/trsc.1050.0135

        a solution that is (start_temp_control_param * 100) % worse will be accepted with a probability 0.5
        """
        obj = carrier.objective()
        acceptance_probability = 0.5
        temperature = -(obj * start_temp_control_param) / log(acceptance_probability)
        return temperature


class PDPTWIteratedLocalSearch(PDPTWMetaHeuristic):
    """
    Uses a perturbation function to explore different regions of the solution space
    """

    def execute(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution,
                carrier_ids: List[int] = None) -> slt.CAHDSolution:
        # FIXME: ILS sometimes returns worse solutions
        # raise NotImplementedError
        solution = deepcopy(solution)
        best_best_solution = solution

        perturbation_num_requests = 2
        random.seed(99)  # to ensure same perturbations in (a) post-acceptance and (b) bidding improvement

        if carrier_ids is None:
            carrier_ids = [x.id_ for x in solution.carriers]

        for carrier_id in carrier_ids:

            self.local_search(instance, solution, [carrier_id])
            best_solution = solution
            self.iter_count = 0
            self.start_time = time.time()

            while not self.stopping_criterion():
                solution_new = self.perturbation(instance, solution, carrier_id, perturbation_num_requests)
                self.local_search(instance, solution_new, [carrier_id])

                if solution_new.objective() > best_solution.objective():
                    best_solution = solution_new

                delta = solution_new.sum_travel_distance() - solution.sum_travel_distance()
                move = (delta, solution, solution_new)
                if self.acceptance_criterion(instance, move):
                    self.update_trajectory('ILS Perturbation', move, True)
                    solution = solution_new  # equivalent to execute_move
                self.iter_count += 1

            if best_solution.objective() > best_best_solution.objective():
                best_best_solution = best_solution

        return best_best_solution

    def acceptance_criterion(self, instance: it.MDPDPTWInstance, move: tuple):
        """
        accept slight degradations: Threshold acceptance

        :param instance:
        :param move:
        :return:
        """
        delta, solution, solution_new = move
        if solution_new.objective() >= solution.objective() * 0.9:
            return True
        else:
            return False

    def perturbation(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution, carrier_id: int,
                     num_requests: int) -> slt.CAHDSolution:
        solution_copy = deepcopy(solution)
        carrier_copy = solution_copy.carriers[carrier_id]
        try:
            # destroy/shake
            # TODO test different shakes
            sh.RandomRemovalShake().execute(instance, carrier_copy, num_requests)

            # repair
            # TODO test different repairs
            cns.MinTravelDistanceInsertion().insert_all_unrouted_statically(instance, solution_copy, carrier_copy.id_)

            return solution_copy

        except ut.ConstraintViolationError:
            # sometimes the shaking cannot be repaired with the given method and will raise a ConstraintViolationError
            # in that case, simply returning the original solution
            return solution

    def local_search(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution, carrier_ids: List[int]):
        """
        improves the solution in place
        """
        # TODO neighborhood should be a parameter instead of an arbitrary choice
        arbitrary_neighborhood = self.neighborhoods[0]
        LocalSearchFirst([arbitrary_neighborhood]).execute(instance, solution, carrier_ids)
        pass

    def stopping_criterion(self):
        if time.time() - self.start_time < TIME_MAX and self.iter_count < ITER_MAX:
            return False
        else:
            return True
