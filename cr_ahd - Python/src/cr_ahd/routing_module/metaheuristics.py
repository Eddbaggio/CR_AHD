import abc
import logging
import random
from copy import deepcopy

from joblib.externals.cloudpickle import instance

from src.cr_ahd.routing_module import local_search as ls
from src.cr_ahd.core_module import instance as it, solution as slt

logger = logging.getLogger(__name__)


class PDPMetaHeuristic:
    def __init__(self):
        self.neighborhoods = [ls.PDPMove,
                              ls.PDPTwoOpt,
                              ls.PDPRelocate,  # inter-tour
                              ]
        self.current_neighborhood = 0
        self.improved = False
        self.stopping_criterion = False
        self.parameters = dict()
        self.history = []

    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        # TODO this was a poor attempt at implementing a general method that works for any metaheuristic
        best_solution = deepcopy(solution)

        for carrier in range(instance.num_carriers):

            neighborhood = self.neighborhoods[self.current_neighborhood]()
            move_generator = neighborhood.feasible_move_generator(instance, solution, carrier)

            self.improved = True

            while self.stopping_criterion is False:
                self.improved = False

                try:
                    move = next(move_generator)
                except StopIteration:  # if no feasible move exists
                    break

                # check tabu
                if self.is_move_tabu(move):
                    continue

                # check acceptance
                accepted = self.acceptance_criterion(instance, solution, carrier, move)
                if accepted:
                    neighborhood.execute_move(instance, solution, carrier, move)
                    logger.info(f'{neighborhood.__class__.__name__} move accepted: {move}')

                    move_generator = neighborhood.feasible_move_generator(instance, solution, carrier)

                    # update the best known solution
                    if solution.sum_profit() > best_solution.sum_profit():
                        best_solution = deepcopy(solution)
                        self.improved = True

                self.update_history(self.current_neighborhood, move, accepted)
                self.update_parameters(move, accepted)
                self.change_neighborhood()
                self.update_stopping_criterion(move, accepted)

        return best_solution

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


class PDPGradientMultiNeighborhoodDescent(PDPMetaHeuristic):
    """
    As a the most basic "metaheuristic". Whether or not solutions are accepted should then happen on this level. Here:
    always accept only improving solutions
    """

    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution):

        for carrier in range(instance.num_carriers):

            for k in range(len(self.neighborhoods)):

                self.current_neighborhood = k
                neighborhood = self.neighborhoods[self.current_neighborhood]()

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
                                move_generator = neighborhood.feasible_move_generator(instance, solution, carrier)

                        except StopIteration:
                            break

        return solution

    def acceptance_criterion(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, move: tuple):
        if move is None:
            return False
        elif move[0] < 0:
            return True
        else:
            return False

    def change_neighborhood(self):
        if self.improved is False and self.current_neighborhood < len(self.neighborhoods):
            self.current_neighborhood += 1
        pass


class PDPVariableNeighborhoodDescent(PDPMetaHeuristic):
    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution):

        for carrier in range(instance.num_carriers):

            self.current_neighborhood = 0

            while self.current_neighborhood < len(self.neighborhoods):
                neighborhood = self.neighborhoods[self.current_neighborhood]()

                all_moves = [move for move in neighborhood.feasible_move_generator(instance, solution, carrier)]
                best_move = min(all_moves, key=lambda x: x[0])

                if self.acceptance_criterion(instance, solution, carrier, best_move):
                    neighborhood.execute_move(instance, solution, carrier, best_move)
                    self.current_neighborhood = 0
                else:
                    self.current_neighborhood += 1

    def acceptance_criterion(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, move: tuple):
        if move[0] < 0:
            return True
        else:
            return False
