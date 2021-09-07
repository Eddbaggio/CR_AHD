import logging.config
import random
from copy import deepcopy
from typing import Union
import time

from src.cr_ahd.utility_module import utils as ut, profiling as pr
from src.cr_ahd.auction_module import auction as au
from src.cr_ahd.core_module import instance as it, solution as slt, tour as tr
from src.cr_ahd.routing_module import tour_construction as cns, tour_initialization as ini, metaheuristics as mh
from src.cr_ahd.tw_management_module import tw_management as twm

logger = logging.getLogger(__name__)


class Solver:
    def __init__(self,
                 tour_construction: cns.PDPParallelInsertionConstruction,
                 tour_improvement: mh.PDPTWMetaHeuristic,
                 time_window_management: twm.TWManagement,
                 auction: Union[au.Auction, bool]
                 ):
        assert isinstance(auction, au.Auction) or auction is False
        self.tour_construction = tour_construction
        self.time_window_management = time_window_management
        self.tour_improvement = tour_improvement
        self.auction = auction

    def execute(self, instance: it.MDPDPTWInstance,
                # starting_solution: slt.CAHDSolution = None
                ):
        """
        apply the concrete solution algorithm
        """

        '''
        # using a starting solution messed up the solver config that is stored with that solution
        if starting_solution is None:
            solution = slt.CAHDSolution(instance)
        else:
            solution = starting_solution
        '''
        solution = slt.CAHDSolution(instance)

        self.update_solution_solver_config(solution)

        logger.info(f'{instance.id_}: Solving {solution.solver_config}')

        random.seed(0)
        timer = pr.Timer()
        instance, solution = self._acceptance_phase(instance, solution)
        timer.write_duration_to_solution(solution, 'runtime_acceptance_phase')

        timer = pr.Timer()
        solution = self._improvement_phase(instance, solution)  # post-acceptance optimization
        timer.write_duration_to_solution(solution, 'runtime_post_acceptance_improvement')

        if self.auction:
            solution = self._auction_phase(instance, solution)

        logger.info(f'{instance.id_}: Success {solution.solver_config}')

        return solution

    def update_solution_solver_config(self, solution):
        solution.solver_config['tour_construction'] = self.tour_construction.__class__.__name__
        solution.solver_config['tour_improvement'] = self.tour_improvement.name
        # solution.solver_config['time_window_management'] = self.time_window_management.__class__.__name__
        solution.solver_config[
            'time_window_offering'] = self.time_window_management.time_window_offering.__class__.__name__
        solution.solver_config[
            'time_window_selection'] = self.time_window_management.time_window_selection.__class__.__name__

        if self.auction:
            solution.solver_config['solution_algorithm'] = 'CollaborativePlanning'
            solution.solver_config['auction_tour_construction'] = self.tour_construction.__class__.__name__
            solution.solver_config['auction_tour_improvement'] = self.tour_improvement.name
            solution.solver_config['num_submitted_requests'] = self.auction.request_selection.num_submitted_requests
            solution.solver_config['request_selection'] = self.auction.request_selection.__class__.__name__
            solution.solver_config['bundle_generation'] = self.auction.bundle_generation.__class__.__name__
            try:
                # for bundle generation with LimitedBundlePoolGenerationBehavior
                solution.solver_config[
                    'bundling_valuation'] = self.auction.bundle_generation.bundling_valuation.__class__.__name__
            except KeyError:
                None
            solution.solver_config['num_auction_bundles'] = self.auction.bundle_generation.num_auction_bundles
            solution.solver_config['bidding'] = self.auction.bidding.__class__.__name__
            solution.solver_config['winner_determination'] = self.auction.winner_determination.__class__.__name__
        else:
            solution.solver_config['solution_algorithm'] = 'IsolatedPlanning'

    def _acceptance_phase(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution):
        instance = deepcopy(instance)
        solution = deepcopy(solution)
        while solution.unassigned_requests:
            # assign the next request
            request = solution.unassigned_requests[0]
            carrier_id = instance.request_to_carrier_assignment[request]
            carrier = solution.carriers[carrier_id]
            solution.assign_requests_to_carriers([request], [carrier_id])

            # find the tw for the request
            accepted = self.time_window_management.execute(instance, solution, carrier, request)

            # build tours with the assigned request if it was accepted
            if accepted:

                self.tour_construction.insert_single(instance, solution, carrier, request)

        ut.validate_solution(instance, solution)  # check to make sure everything's functional
        return instance, solution

    def _improvement_phase(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution):
        before = solution.objective()
        solution = self.tour_improvement.execute(instance, solution)
        ut.validate_solution(instance, solution)
        assert solution.objective() >= before
        return solution

    def _static_routing(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution):
        raise NotImplementedError('static routing is omitted atm since it does not yield improvements over the '
                                  'dynamic routes or fails with infeasibility')
        solution = deepcopy(solution)
        solution.clear_carrier_routes()

        # create seed tours
        ini.MaxCliqueTourInitialization().execute(instance, solution)

        # construct_static initial solution
        self.tour_construction.insert_all(instance, solution, carrier)

        ut.validate_solution(instance, solution)
        return solution

    def _auction_phase(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution):
        """
        includes request selection, bundle generation, bidding, winner determination and also the final routing
        after the auction
        :param instance:
        :param solution:
        :return:
        """
        solution = self.auction.execute_auction(instance, solution)
        return solution