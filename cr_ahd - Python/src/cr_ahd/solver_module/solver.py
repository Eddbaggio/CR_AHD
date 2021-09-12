import logging.config
import random
from copy import deepcopy
from typing import Tuple

from src.cr_ahd.auction_module import auction as au
from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import metaheuristics as mh
from src.cr_ahd.routing_module import tour_construction as cns
from src.cr_ahd.tw_management_module import tw_offering as two, tw_selection as tws
from src.cr_ahd.utility_module import utils as ut, profiling as pr

logger = logging.getLogger(__name__)


class Solver:
    def __init__(self,
                 time_window_offering: two.TWOfferingBehavior,
                 time_window_selection: tws.TWSelectionBehavior,
                 tour_construction: cns.PDPParallelInsertionConstruction,
                 tour_improvement: mh.PDPTWMetaHeuristic,
                 num_intermediate_auctions: int = 0,
                 intermediate_auction: au.Auction = False,
                 final_auction: au.Auction = False,
                 ):
        assert not (bool(num_intermediate_auctions) ^ bool(intermediate_auction))  # not XOR
        self.time_window_offering: two.TWOfferingBehavior = time_window_offering
        self.time_window_selection: tws.TWSelectionBehavior = time_window_selection
        self.tour_construction: cns.PDPParallelInsertionConstruction = tour_construction
        self.tour_improvement: mh.PDPTWMetaHeuristic = tour_improvement
        self.num_intermediate_auctions: int = num_intermediate_auctions
        self.intermediate_auction: au.Auction = intermediate_auction
        self.final_auction: au.Auction = final_auction

    def execute(self,
                instance: it.MDPDPTWInstance,
                starting_solution: slt.CAHDSolution = None
                ) -> Tuple[it.MDPDPTWInstance, slt.CAHDSolution]:
        """
        apply the concrete solution algorithm
        """

        # ===== [0] Setup =====
        instance = deepcopy(instance)
        if starting_solution is None:
            solution = slt.CAHDSolution(instance)
        else:
            solution = starting_solution
            solution.timings.clear()

        self.update_solution_solver_config(
            solution)  # TODO reverse this. it should be a method of the solution not the solver
        random.seed(0)
        logger.info(f'{instance.id_}: Solving {solution.solver_config}')

        # ======================
        # ===== NEW SOLVER =====
        # ======================
        i = 0
        while solution.unassigned_requests:
            # ===== [1] Dynamic Acceptance Phase =====
            for request in range(i, instance.num_requests, instance.num_requests_per_carrier):

                assert request in solution.unassigned_requests
                carrier_id = instance.request_to_carrier_assignment[request]
                carrier = solution.carriers[carrier_id]
                solution.assign_requests_to_carriers([request], [carrier_id])

                # ===== [2] Time Window Management =====
                timer = pr.Timer()
                offer_set = self.time_window_offering.execute(instance, carrier, request)  # which TWs to offer?
                selected_tw = self.time_window_selection.execute(offer_set, request)  # which TW is selected?
                timer.write_duration_to_solution(solution, 'runtime_time_window_management', True)

                pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(request)
                instance.assign_time_window(pickup_vertex, ut.TIME_HORIZON)

                if selected_tw:
                    instance.assign_time_window(delivery_vertex, selected_tw)
                    carrier.accepted_requests.append(request)
                    carrier.acceptance_rate = len(carrier.accepted_requests) / len(carrier.assigned_requests)
                    # ===== [3] Dynamic Routing/Insertion =====
                    timer = pr.Timer()
                    self.tour_construction.insert_single_request(instance, solution, carrier.id_, request)
                    timer.write_duration_to_solution(solution, 'runtime_dynamic_insertion', True)

                else:  # in case (a) the offer set was empty or (b) the customer did not select any time window
                    logger.error(f'[{instance.id_}] No feasible TW can be offered '
                                 f'from Carrier {carrier.id_} to request {request}')
                    instance.assign_time_window(delivery_vertex, ut.TIME_HORIZON)
                    carrier.rejected_requests.append(request)
                    carrier.unrouted_requests.pop(0)
                    carrier.acceptance_rate = len(carrier.accepted_requests) / len(carrier.assigned_requests)

            # ===== [4] Intermediate Auctions =====
            if self.intermediate_auction:
                if ((i + 1) * instance.num_carriers) % (
                        instance.num_requests / (self.num_intermediate_auctions + 1)) == 0:
                    timer = pr.Timer()
                    solution = self.tour_improvement.execute(instance, solution)
                    timer.write_duration_to_solution(solution, 'runtime_intermediate_improvements', True)
                    timer = pr.Timer()
                    solution = self.intermediate_auction.execute(instance, solution)
                    timer.write_duration_to_solution(solution, 'runtime_intermediate_auctions', True)

            i += 1

        # ===== [5] Final Improvement =====
        timer = pr.Timer()
        solution = self.tour_improvement.execute(instance, solution)
        timer.write_duration_to_solution(solution, 'runtime_final_improvement')

        # ===== [6] Final Auction =====
        if self.final_auction:
            timer = pr.Timer()
            solution = self.final_auction.execute(instance, solution)
            timer.write_duration_to_solution(solution, 'runtime_final_auction')

        ut.validate_solution(instance, solution)  # safety check to make sure everything's functional
        logger.info(f'{instance.id_}: Success {solution.solver_config}')

        return instance, solution

    def update_solution_solver_config(self, solution):  # TODO this should be a method of the solution, not the solver!
        """
        The solver config describes the solution methods and used to solve an instance. For post-processing and
        comparing solutions it is thus useful to store the methods' names with the solution.
        """

        config = solution.solver_config
        int_auction = self.intermediate_auction
        fin_auction = self.final_auction
        if int_auction or fin_auction:
            config['solution_algorithm'] = 'CollaborativePlanning'
        else:
            config['solution_algorithm'] = 'IsolatedPlanning'

        config['tour_construction'] = self.tour_construction.name
        config['tour_improvement'] = self.tour_improvement.name
        config['time_window_offering'] = self.time_window_offering.name
        config['time_window_selection'] = self.time_window_selection.name
        config['num_int_auctions'] = self.num_intermediate_auctions

        if int_auction:
            config['int_auction_tour_construction'] = int_auction.tour_construction.name
            config['int_auction_tour_improvement'] = int_auction.tour_improvement.name
            config['int_auction_num_submitted_requests'] = int_auction.request_selection.num_submitted_requests
            config['int_auction_request_selection'] = int_auction.request_selection.name
            config['int_auction_bundle_generation'] = int_auction.bundle_generation.name
            try:
                # for bundle generation with LimitedBundlePoolGenerationBehavior
                config['int_auction_bundling_valuation'] = int_auction.bundle_generation.bundling_valuation.name
            except KeyError:
                None
            config['int_auction_num_auction_bundles'] = int_auction.bundle_generation.num_auction_bundles
            config['int_auction_bidding'] = int_auction.bidding.name
            config['int_auction_winner_determination'] = int_auction.winner_determination.name
            config['int_auction_num_auction_rounds'] = int_auction.num_auction_rounds

        if fin_auction:
            config['fin_auction_tour_construction'] = fin_auction.tour_construction.name
            config['fin_auction_tour_improvement'] = fin_auction.tour_improvement.name
            config['fin_auction_num_submitted_requests'] = fin_auction.request_selection.num_submitted_requests
            config['fin_auction_request_selection'] = fin_auction.request_selection.name
            config['fin_auction_bundle_generation'] = fin_auction.bundle_generation.name
            try:
                # for bundle generation with LimitedBundlePoolGenerationBehavior
                config['fin_auction_bundling_valuation'] = fin_auction.bundle_generation.bundling_valuation.name
            except KeyError:
                None
            config['fin_auction_num_auction_bundles'] = fin_auction.bundle_generation.num_auction_bundles
            config['fin_auction_bidding'] = fin_auction.bidding.name
            config['fin_auction_winner_determination'] = fin_auction.winner_determination.name
            config['fin_auction_num_auction_rounds'] = fin_auction.num_auction_rounds

        pass

    """
    def _static_routing(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution):
        raise NotImplementedError('static routing is omitted atm since it does not yield improvements over the '
                                  'dynamic routes or fails with infeasibility')
        solution = deepcopy(solution)
        solution.clear_carrier_routes()

        # create seed tours
        ini.MaxCliqueTourInitialization().execute(instance, solution)

        # construct_static initial solution
        self.tour_construction.insert_all_requests(instance, solution, carrier)

        ut.validate_solution(instance, solution)
        return solution
    """
