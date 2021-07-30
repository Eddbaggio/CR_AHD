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

    @pr.timing
    def execute(self, instance: it.PDPInstance, starting_solution: slt.CAHDSolution = None):
        """
        apply the concrete solution algorithm
        """

        if starting_solution is None:
            solution = slt.CAHDSolution(instance)
        else:
            solution = starting_solution

        self.update_solution_solver_config(solution)

        logger.info(f'{instance.id_}: Solving {solution.solver_config}')

        random.seed(0)

        solution = self._acceptance_phase(instance, solution)
        # if self.tour_improvement:
        solution = self._improvement_phase(instance, solution)  # post-acceptance optimization
        if self.auction:
            solution = self._auction_phase(instance, solution)

        logger.info(f'{instance.id_}: Success {solution.solver_config}')

        return solution

    def update_solution_solver_config(self, solution):
        if self.auction:
            solution.solver_config['solution_algorithm'] = 'CollaborativePlanning'
            solution.solver_config['auction_tour_construction'] = self.tour_construction.__class__.__name__
            solution.solver_config['auction_tour_improvement'] = self.tour_improvement.__class__.__name__
            solution.solver_config['num_submitted_requests'] = self.auction.request_selection.num_submitted_requests
            solution.solver_config['request_selection'] = self.auction.request_selection.__class__.__name__
            solution.solver_config['bundle_generation'] = self.auction.bundle_generation.__class__.__name__
            try:
                # for GA-based bundle generation
                solution.solver_config['bundling_valuation'] = self.auction.bundle_generation.parameters[
                    'bundling_valuation'].__class__.__name__
            except KeyError:
                None
            solution.solver_config['num_auction_bundles'] = self.auction.bundle_generation.num_auction_bundles
            solution.solver_config['bidding'] = self.auction.bidding.__class__.__name__
            solution.solver_config['winner_determination'] = self.auction.winner_determination.__class__.__name__
        else:
            solution.solver_config['solution_algorithm'] = 'IsolatedPlanning'

        solution.solver_config['tour_construction'] = self.tour_construction.__class__.__name__
        solution.solver_config['tour_improvement'] = self.tour_improvement.__class__.__name__
        solution.solver_config['time_window_management'] = self.time_window_management.__class__.__name__
        solution.solver_config[
            'time_window_offering'] = self.time_window_management.time_window_offering.__class__.__name__
        solution.solver_config[
            'time_window_selection'] = self.time_window_management.time_window_selection.__class__.__name__

    @pr.timing
    def _acceptance_phase(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        solution = deepcopy(solution)
        while solution.unassigned_requests:
            # assign the next request
            request = solution.unassigned_requests[0]
            carrier = instance.request_to_carrier_assignment[request]
            solution.assign_requests_to_carriers([request], [carrier])

            # find the tw for the request
            accepted = self.time_window_management.execute(instance, solution, carrier, request)

            # build tours with the assigned request if it was accepted
            if accepted:
                self.tour_construction.construct_dynamic(instance, solution, carrier)

        ut.validate_solution(instance, solution)
        return solution

    @pr.timing
    def _improvement_phase(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        solution = deepcopy(solution)
        self.tour_improvement.execute(instance, solution)
        return solution

    def _static_routing(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        raise NotImplementedError('static routing is omitted atm since it does not yield improvements over the '
                                  'dynamic routes or fails with infeasibility')
        solution = deepcopy(solution)
        solution.clear_carrier_routes()

        # create seed tours
        ini.MaxCliqueTourInitialization().execute(instance, solution)

        # construct_static initial solution
        self.tour_construction.construct_static(instance, solution, carrier)

        ut.validate_solution(instance, solution)
        return solution

    @pr.timing
    def _auction_phase(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        """
        includes request selection, bundle generation, bidding, winner determination and also the final routing
        after the auction
        :param instance:
        :param solution:
        :return:
        """
        solution = deepcopy(solution)
        self.auction.execute_auction(instance, solution)
        return solution


class CentralizedPlanning(Solver):

    def execute(self, instance: it.PDPInstance, starting_solution: slt.CAHDSolution = None):
        # copy and alter the underlying instance to make it a multi-depot, single-carrier instance
        md_instance = deepcopy(instance)
        md_instance.num_carriers = 1
        md_instance.carrier_depots = [[d for d in range(instance.num_depots)]]
        md_instance.request_to_carrier_assignment = [0] * len(md_instance.request_to_carrier_assignment)

        # copy, clear and adjust the solution, in particular the TW will be maintained from the starting solution
        solution = self._make_centralized_solution(md_instance, starting_solution)

        random.seed(0)

        # do dynamic insertion
        carrier_ = solution.carriers[0]
        while carrier_.unrouted_requests:
            # carrier will always be 0 for centralized
            cns.MinTravelDistanceInsertion().construct_dynamic(instance, solution, 0)

        solution = self._improvement_phase(md_instance, solution)

        solution.solution_algorithm = self.__class__.__name__
        return solution

    def _make_centralized_solution(self, md_instance, starting_solution):
        solution = deepcopy(starting_solution)
        solution.clear_carrier_routes()
        central_carrier_ = slt.AHDSolution(0)
        central_carrier_.assigned_requests = [request for carrier_ in solution.carriers for request in
                                              carrier_.assigned_requests]
        central_carrier_.accepted_requests = [request for carrier_ in solution.carriers for request in
                                              carrier_.accepted_requests]
        central_carrier_.rejected_requests = [request for carrier_ in solution.carriers for request in
                                              carrier_.rejected_requests]
        central_carrier_.routed_requests = []
        central_carrier_.unrouted_requests = [request for carrier_ in solution.carriers for request in
                                              carrier_.unrouted_requests]
        central_carrier_.acceptance_rate = len(central_carrier_.accepted_requests) / len(
            central_carrier_.assigned_requests)
        solution.carriers.clear()
        solution.carriers.append(central_carrier_)
        solution.carrier_depots = [[depot for depot in range(md_instance.num_depots)]]
        solution.request_to_carrier_assignment = [0] * len(solution.request_to_carrier_assignment)
        solution.solution_algorithm = None
        return solution

    """
    def _acceptance_phase(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        solution = deepcopy(solution)
        while solution.unassigned_requests:
            # assign the next request
            request = solution.unassigned_requests[0]
            carrier = instance.request_to_carrier_assignment[request]
            solution.assign_requests_to_carriers([request], [carrier])

            # find the tw for the request
            accepted = self._time_window_management(instance, solution, carrier, request)
            construction = cns.MinTravelDistanceInsertion()
            # insert assigned request if it was accepted
            if accepted:
                insertion_criteria, tour, pickup_pos, delivery_pos = construction.best_insertion_for_request(instance,
                                                                                                             solution,
                                                                                                             carrier,
                                                                                                             request)

                if tour is None:
                    # print(f'{request}: new tour')
                    self._create_new_tour_with_request(instance, solution, carrier, request)
                else:
                    # print(f'{request}: t{solution.carriers[carrier].tours[best_tour].routing_sequence[0]} '
                    #       f'{best_pickup_pos, best_delivery_pos}')
                    construction.execute_insertion(instance, solution, carrier, request, tour, pickup_pos, delivery_pos)

        ut.validate_solution(instance, solution)
        return solution
        """

    def _create_new_tour_with_request(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int,
                                      request: int):
        """only for multi-depot instances. ensures that non-routable requests will be assigned to their original
        depot """
        carrier_ = solution.carriers[carrier]
        if carrier_.num_tours() >= instance.carriers_max_num_tours * len(solution.carrier_depots[carrier]):
            # logger.error(f'Max Vehicle Constraint violated!')
            raise ut.ConstraintViolationError(f'Cannot create new route with request {request} for carrier {carrier}.'
                                              f' Max. number of vehicles is {instance.carriers_max_num_tours}!'
                                              f' ({instance.id_})')

        original_depot = request // instance.num_requests_per_carrier
        tour_ = tr.Tour(carrier_.num_tours(), instance, solution, depot_index=original_depot)
        if tour_.insertion_feasibility_check(instance, solution, [1, 2], instance.pickup_delivery_pair(request)):
            tour_.insert_and_update(instance, solution, [1, 2], instance.pickup_delivery_pair(request))
        else:
            raise ut.ConstraintViolationError(f'Cannot create new route with request {request} for carrier {carrier}.')

        carrier_.tours.append(tour_)
        carrier_.unrouted_requests.remove(request)
        carrier_.routed_requests.append(request)
        return

    def _time_window_management(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, request: int):
        # use only the single, centralized carrier
        carrier = 0
        return twm.TWManagementSingleOriginalDepot().execute(instance, solution, carrier, request)



