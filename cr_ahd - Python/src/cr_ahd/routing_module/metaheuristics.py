import abc
import logging
import random
from copy import deepcopy
from math import exp, log

from src.cr_ahd.routing_module import local_search as ls
from src.cr_ahd.core_module import instance as it, solution as slt

logger = logging.getLogger(__name__)


class PDPMetaHeuristic(abc.ABC):
    def __init__(self):
        self.neighborhoods = [ls.PDPMove,
                              ls.PDPTwoOpt,
                              # ls.PDPRelocate,  # inter-tour
                              # ls.PDPRelocate2,  # inter-tour
                              ]
        self.improved = False
        self.stopping_criterion = False
        self.parameters = dict()
        self.history = []  # collection of e.g. visited neighbors, accepted moves, ...
        self.trajectory = []  # collection of all accepted & executed moves

    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carriers=None):
        pass
        # TODO this was a poor attempt at implementing a general method that works for any metaheuristic
        # best_solution = deepcopy(solution)
        #
        # for carrier in range(instance.num_carriers):
        #
        #     neighborhood = self.neighborhoods[self.current_neighborhood]()
        #     move_generator = neighborhood.feasible_move_generator(instance, solution, carrier)
        #
        #     self.improved = True
        #
        #     while self.stopping_criterion is False:
        #         self.improved = False
        #
        #         try:
        #             move = next(move_generator)
        #         except StopIteration:  # if no feasible move exists
        #             break
        #
        #         # check tabu
        #         if self.is_move_tabu(move):
        #             continue
        #
        #         # check acceptance
        #         accepted = self.acceptance_criterion(instance, solution, carrier, move)
        #         if accepted:
        #             neighborhood.execute_move(instance, solution, carrier, move)
        #             logger.info(f'{neighborhood.__class__.__name__} move accepted: {move}')
        #
        #             move_generator = neighborhood.feasible_move_generator(instance, solution, carrier)
        #
        #             # update the best known solution
        #             if solution.sum_profit() > best_solution.sum_profit():
        #                 best_solution = deepcopy(solution)
        #                 self.improved = True
        #
        #         self.update_history(self.current_neighborhood, move, accepted)
        #         self.update_parameters(move, accepted)
        #         self.change_neighborhood()
        #         self.update_stopping_criterion(move, accepted)
        #
        # return best_solution

    # @abc.abstractmethod
    def acceptance_criterion(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, move: tuple):
        return True

    # @abc.abstractmethod
    def change_neighborhood(self):
        pass

    # @abc.abstractmethod
    def update_stopping_criterion(self, move: tuple, accepted: bool):
        pass

    def update_parameters(self, move: tuple, accepted: bool):
        pass

    def is_move_tabu(self, move: tuple):
        return False

    def update_history(self, k, move, accepted):
        self.history.append((self.neighborhoods[k].__name__, move, accepted))
        pass


class NoMetaheuristic(PDPMetaHeuristic):
    """Placeholder for cases in which no improvement is wanted"""

    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carriers=None):
        pass


class PDPSequentialNeighborhoodDescent(PDPMetaHeuristic):
    """
    Sequentially exhaust each neighborhood. Only improvements are accepted
    """

    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carriers=None):
        if carriers is None:
            carriers = range(len(solution.carriers))

        for carrier in carriers:
            for k in range(len(self.neighborhoods)):
                neighborhood = self.neighborhoods[k]()
                move_generator = neighborhood.feasible_move_generator(instance, solution, carrier)
                self.improved = True
                while self.improved:
                    self.improved = False
                    accepted = False
                    while accepted is False:
                        try:
                            move = next(move_generator)
                            if self.acceptance_criterion(instance, solution, carrier, move):
                                accepted = True
                                neighborhood.execute_move(instance, solution, carrier, move)
                                self.improved = True
                                self.trajectory.append(move)
                                move_generator = neighborhood.feasible_move_generator(instance, solution, carrier)
                        except StopIteration:
                            break
        pass

    def acceptance_criterion(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, move: tuple):
        if move is None:
            return False
        elif move[0] < 0:
            return True
        else:
            return False


class PDPRandomNeighborhoodDescent(PDPMetaHeuristic):
    """
    randomly select a neighborhood for the next improving move
    """
    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carriers=None):
        raise NotImplementedError
        if carriers is None:
            carriers = range(len(solution.carriers))

        for carrier in carriers:

            # start in a random neighborhood
            eligible_neighborhoods = list(range(len(self.neighborhoods)))
            k = random.choice(eligible_neighborhoods)

            while eligible_neighborhoods:
                neighborhood = self.neighborhoods[k]()
                all_moves = [move for move in neighborhood.feasible_move_generator(instance, solution, carrier)]
                if any(all_moves):
                    best_move = min(all_moves, key=lambda x: x[0])
                    # if a move is accepted in the current neighborhood
                    if self.acceptance_criterion(instance, solution, carrier, best_move):
                        neighborhood.execute_move(instance, solution, carrier, best_move)
                        self.trajectory.append(best_move)
                        k = 0
                    else:
                        eligible_neighborhoods.remove(k)
                        k = random.choice(eligible_neighborhoods)
                else:
                    k += 1
        pass

    def acceptance_criterion(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, move: tuple):
        if move[0] < 0:
            return True
        else:
            return False


class PDPVariableNeighborhoodDescent(PDPMetaHeuristic):
    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carriers=None):
        if carriers is None:
            carriers = range(len(solution.carriers))

        for carrier in carriers:
            k = 0
            while k < len(self.neighborhoods):
                neighborhood = self.neighborhoods[k]()
                all_moves = [move for move in neighborhood.feasible_move_generator(instance, solution, carrier)]
                if any(all_moves):
                    best_move = min(all_moves, key=lambda x: x[0])
                    if self.acceptance_criterion(instance, solution, carrier, best_move):
                        neighborhood.execute_move(instance, solution, carrier, best_move)
                        self.trajectory.append(best_move)
                        k = 0
                    else:
                        k += 1
                else:
                    k += 1
        pass

    def acceptance_criterion(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, move: tuple):
        if move[0] < 0:
            return True
        else:
            return False


class PDPVariableNeighborhoodDescentFirst(PDPMetaHeuristic):
    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carriers=None):
        if carriers is None:
            carriers = range(len(solution.carriers))

        for carrier in carriers:
            k = 0
            while k < len(self.neighborhoods):
                neighborhood = self.neighborhoods[k]()
                move_generator = neighborhood.feasible_move_generator(instance, solution, carrier)
                try:
                    while True:
                        move = next(move_generator)
                        if self.acceptance_criterion(instance, solution, carrier, move):
                            neighborhood.execute_move(instance, solution, carrier, move)
                            self.trajectory.append(move)
                            k = 0
                            break
                except StopIteration:
                    k += 1
        pass

    def acceptance_criterion(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, move: tuple):
        if move[0] < 0:
            return True
        else:
            return False


class PDPSimulatedAnnealing(PDPMetaHeuristic):
    def __init__(self):
        super().__init__()
        self.parameters['initial_temp'] = None
        self.parameters['temp'] = None
        self.parameters['max_iterations'] = 50  # TODO adjust these two params
        self.parameters['max_iteration_per_temperature'] = 15
        self.parameters['alpha'] = 0.8  # for cooling schedule (usually 0.8 <= alpha <=0.9)

    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carriers=None):
        if carriers is None:
            carriers = range(len(solution.carriers))
        raise NotImplementedError(
            'Metaheuristics should modify the solution in place. This has not yet been fixed for SA')
        best_solution = deepcopy(solution)
        tentative_solution = deepcopy(solution)

        for carrier in carriers:
            self.compute_start_temperature(tentative_solution, carrier)
            k = random.randint(0, len(self.neighborhoods) - 1)  # random neighborhood
            neighborhood = self.neighborhoods[k]()
            move_generator = neighborhood.feasible_move_generator(instance, tentative_solution, carrier)
            i = 0
            while i < self.parameters['max_iterations']:
                # adjust temperature
                self.parameters['temp'] = self.parameters['initial_temp'] * self.parameters['alpha'] ** i
                # try different neighbors
                try:
                    m = 0
                    while m < self.parameters['max_iteration_per_temperature']:
                        # with or without replacement? without since the same move might be accepted next time
                        move = random.choice(list(move_generator))  # random neighbor
                        accepted = self.acceptance_criterion(instance, tentative_solution, carrier, move)
                        if accepted:
                            neighborhood.execute_move(instance, tentative_solution, carrier, move)
                            self.trajectory.append(move)
                            move_generator = neighborhood.feasible_move_generator(instance, tentative_solution, carrier)
                            if tentative_solution.sum_profit() >= best_solution.sum_profit():
                                best_solution = deepcopy(tentative_solution)
                        m += 1
                except IndexError:  # if no move exists
                    break

                i += 1

        return best_solution

    def compute_start_temperature(self, solution, carrier: int, start_temp_control_param=0.5):
        """compute start temperature according to Ropke,S., & Pisinger,D. (2006). An Adaptive Large Neighborhood
        Search Heuristic for the Pickup and Delivery Problem with Time Windows. Transportation Science, 40(4),
        455–472. https://doi.org/10.1287/trsc.1050.0135

        a solution that is (start_temp_control_param * 100) % worse will be accepted with a probability 0.5
        """
        carrier_ = solution.carriers[carrier]
        obj = carrier_.sum_profit()
        acceptance_probability = 0.5
        self.parameters['initial_temp'] = -(obj * start_temp_control_param) / log(acceptance_probability)
        self.parameters['temp'] = self.parameters['initial_temp']

    def acceptance_criterion(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, move: tuple):
        if move is None:
            return False
        elif move[0] <= 0:
            return True
        elif random.random() < exp(-move[0] / self.parameters['temp']):
            return True
        else:
            return False
