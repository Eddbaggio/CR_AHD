import logging.config
import random
from copy import deepcopy
from typing import Tuple
import numpy as np

from auction_module import auction as au
from core_module import instance as it, solution as slt
from routing_module import metaheuristics as mh
from routing_module import tour_construction as cns
from tw_management_module import tw_offering as two, tw_selection as tws
from utility_module import utils as ut, profiling as pr

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

        self.config = {
            'solution_algorithm': 'CollaborativePlanning' if intermediate_auction or final_auction else 'IsolatedPlanning',
            'tour_improvement': tour_improvement.name,
            'neighborhoods': '+'.join([nbh.name for nbh in tour_improvement.neighborhoods]),
            'tour_construction': tour_construction.name,
            'tour_improvement_time_limit_per_carrier': tour_improvement.time_limit_per_carrier,
            'time_window_offering': time_window_offering.name,
            'time_window_selection': time_window_selection.name,
            'num_int_auctions': num_intermediate_auctions,

            'int_auction_tour_construction': None,
            'int_auction_tour_improvement': None,
            'int_auction_neighborhoods': None,
            'int_auction_num_submitted_requests': None,
            'int_auction_request_selection': None,
            'int_auction_bundle_generation': None,
            'int_auction_bundling_valuation': None,
            'int_auction_num_auction_bundles': None,
            'int_auction_bidding': None,
            'int_auction_winner_determination': None,
            'int_auction_num_auction_rounds': None,
            'fin_auction_tour_construction': None,
            'fin_auction_tour_improvement': None,
            'fin_auction_neighborhoods': None,
            'fin_auction_num_submitted_requests': None,
            'fin_auction_request_selection': None,
            'fin_auction_bundle_generation': None,
            'fin_auction_bundling_valuation': None,
            'fin_auction_num_auction_bundles': None,
            'fin_auction_bidding': None,
            'fin_auction_winner_determination': None,
            'fin_auction_num_auction_rounds': None,
        }

        if intermediate_auction:
            self.config.update({
                'int_auction_tour_construction': intermediate_auction.tour_construction.name,
                'int_auction_tour_improvement': intermediate_auction.tour_improvement.name,
                'int_auction_neighborhoods': '+'.join(
                    [nbh.name for nbh in intermediate_auction.tour_improvement.neighborhoods]),
                'int_auction_num_submitted_requests': intermediate_auction.request_selection.num_submitted_requests,
                'int_auction_request_selection': intermediate_auction.request_selection.name,
                'int_auction_bundle_generation': intermediate_auction.bundle_generation.name,
                # FIXME not all bundle_generation approaches have a bundling_valuation!
                'int_auction_bundling_valuation': intermediate_auction.bundle_generation.bundling_valuation.name,
                'int_auction_num_auction_bundles': intermediate_auction.bundle_generation.num_auction_bundles,
                'int_auction_bidding': intermediate_auction.bidding.name,
                'int_auction_winner_determination': intermediate_auction.winner_determination.name,
                'int_auction_num_auction_rounds': intermediate_auction.num_auction_rounds,
            })

        if final_auction:
            self.config.update({
                'fin_auction_tour_construction': final_auction.tour_construction.name,
                'fin_auction_tour_improvement': final_auction.tour_improvement.name,
                'fin_auction_neighborhoods': '+'.join(
                    [nbh.name for nbh in final_auction.tour_improvement.neighborhoods]),
                'fin_auction_num_submitted_requests': final_auction.request_selection.num_submitted_requests,
                'fin_auction_request_selection': final_auction.request_selection.name,
                'fin_auction_bundle_generation': final_auction.bundle_generation.name,
                # FIXME not all bundle_generation approaches have a bundling_valuation!
                'fin_auction_bundling_valuation': final_auction.bundle_generation.bundling_valuation.name,
                'fin_auction_num_auction_bundles': final_auction.bundle_generation.num_auction_bundles,
                'fin_auction_bidding': final_auction.bidding.name,
                'fin_auction_winner_determination': final_auction.winner_determination.name,
                'fin_auction_num_auction_rounds': final_auction.num_auction_rounds,
            })

    def execute(self,
                instance: it.MDPDPTWInstance,
                starting_solution: slt.CAHDSolution = None
                ) -> Tuple[it.MDPDPTWInstance, slt.CAHDSolution]:
        """
        apply the concrete solution algorithm
        """

        # ===== [0] Setup =====
        if self.intermediate_auction:
            # define the iterations at which an intermediate auction shall take place
            intermediate_auction_timepoints = [
                round(x) for x in np.linspace(-1,
                                              instance.num_requests_per_carrier - 1,
                                              self.num_intermediate_auctions + 1,
                                              False)[1:]
            ]

        instance = deepcopy(instance)
        if starting_solution is None:
            solution = slt.CAHDSolution(instance)
        else:
            solution = starting_solution
            solution.timings.clear()

        solution.solver_config.update(self.config)
        random.seed(0)
        logger.info(f'{instance.id_}: Solving {solution.solver_config}')

        i = 0
        while solution.unassigned_requests:
            # ===== [1] Dynamic Acceptance Phase =====
            for request in range(i, instance.num_requests, instance.num_requests_per_carrier):

                assert request in solution.unassigned_requests
                carrier_id = instance.request_to_carrier_assignment[request]
                carrier = solution.carriers[carrier_id]
                solution.assign_requests_to_carriers([request], [carrier_id])

                # ===== [2] Time Window Management =====
                if self.time_window_offering and self.time_window_selection:
                    timer = pr.Timer()
                    offer_set = self.time_window_offering.execute(instance, carrier, request)  # which TWs to offer?
                    selected_tw = self.time_window_selection.execute(offer_set, request)  # which TW is selected?
                    timer.write_duration_to_solution(solution, 'runtime_time_window_management', True)

                    pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(request)
                    instance.assign_time_window(pickup_vertex, ut.EXECUTION_TIME_HORIZON)

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
                        instance.assign_time_window(delivery_vertex, ut.EXECUTION_TIME_HORIZON)
                        carrier.rejected_requests.append(request)
                        carrier.unrouted_requests.pop(0)
                        carrier.acceptance_rate = len(carrier.accepted_requests) / len(carrier.assigned_requests)

            # ===== [4] Intermediate Auctions =====
            if self.intermediate_auction:
                if i in intermediate_auction_timepoints:
                    timer = pr.Timer()
                    # solution = self.tour_improvement.execute(instance, solution)
                    timer.write_duration_to_solution(solution, 'runtime_intermediate_improvements', True)
                    timer = pr.Timer()
                    solution = self.intermediate_auction.execute(instance, solution)
                    timer.write_duration_to_solution(solution, 'runtime_intermediate_auctions', True)

            i += 1

        # ===== [5] Final Improvement =====
        before_improvement = solution.objective()
        timer = pr.Timer()
        solution = self.tour_improvement.execute(instance, solution)
        timer.write_duration_to_solution(solution, 'runtime_final_improvement')
        assert int(before_improvement) <= int(solution.objective()), instance.id_
        ut.validate_solution(instance, solution)  # safety check to make sure everything's functional

        # ===== [6] Final Auction =====
        if self.final_auction:
            timer = pr.Timer()
            solution = self.final_auction.execute(instance, solution)
            timer.write_duration_to_solution(solution, 'runtime_final_auction')

        ut.validate_solution(instance, solution)  # safety check to make sure everything's functional
        logger.info(f'{instance.id_}: Success {solution.solver_config}')

        return instance, solution

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
