import datetime as dt
import logging.config
import random
from copy import deepcopy
from typing import Tuple

from auction_module import auction as au
from core_module import instance as it, solution as slt
from routing_module import metaheuristics as mh
from routing_module import tour_construction as cns
from tw_management_module import request_acceptance as ra
from utility_module import utils as ut, profiling as pr
from utility_module.cr_ahd_logging import SUCCESS

logger = logging.getLogger(__name__)


class Solver:
    def __init__(self,
                 request_acceptance: ra.RequestAcceptanceBehavior,
                 tour_construction: cns.VRPTWInsertionConstruction,
                 tour_improvement: mh.VRPTWMetaHeuristic,
                 num_intermediate_auctions: int = 0,
                 intermediate_auction: au.Auction = False,
                 final_auction: au.Auction = False,
                 ):
        assert not (bool(num_intermediate_auctions) ^ bool(intermediate_auction))  # not XOR

        self.request_acceptance: ra.RequestAcceptanceBehavior = request_acceptance
        self.tour_construction: cns.VRPTWInsertionConstruction = tour_construction
        self.tour_improvement: mh.VRPTWMetaHeuristic = tour_improvement
        self.num_intermediate_auctions: int = num_intermediate_auctions
        self.intermediate_auction: au.Auction = intermediate_auction
        self.final_auction: au.Auction = final_auction

        self.config = {
            'solution_algorithm': 'CollaborativePlanning' if intermediate_auction or final_auction else 'IsolatedPlanning',
            'tour_improvement': tour_improvement.name,
            'neighborhoods': '+'.join([nbh.name for nbh in tour_improvement.neighborhoods]),
            'tour_construction': tour_construction.name,
            'tour_improvement_time_limit_per_carrier': tour_improvement.time_limit_per_carrier,
            'max_num_accepted_infeasible': request_acceptance.max_num_accepted_infeasible,
            'request_acceptance_attractiveness': request_acceptance.request_acceptance_attractiveness.name,
            'time_window_offering': request_acceptance.time_window_offering.name,
            'time_window_selection': request_acceptance.time_window_selection.name,
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
                instance: it.MDVRPTWInstance,
                starting_solution: slt.CAHDSolution = None,
                ) -> Tuple[it.MDVRPTWInstance, slt.CAHDSolution]:
        """
        apply the concrete steps of the solution algorithms specified in the config

        :return a tuple of the adjusted instance with its newly assigned time windows and the solution
        """

        # ===== [0] Setup =====
        if self.intermediate_auction:
            intermediate_auction_times = ut.datetime_range(ut.ACCEPTANCE_START_TIME, ut.EXECUTION_START_TIME,
                                                           num=self.num_intermediate_auctions, startpoint=False,
                                                           endpoint=False)
            intermediate_auction_threshold = next(intermediate_auction_times)

        instance = deepcopy(instance)
        if starting_solution is None:
            solution = slt.CAHDSolution(instance)
        else:
            solution = starting_solution
            solution.timings.clear()

        solution.solver_config.update(self.config)
        random.seed(0)
        logger.info(f'{instance.id_}: Solving {solution.solver_config}')

        # ===== Dynamic Request Arrival Phase =====
        for request in sorted(instance.requests, key=lambda x: instance.request_disclosure_time[x]):

            # ===== Intermediate Auctions =====
            if self.intermediate_auction:
                if instance.request_disclosure_time[request] >= intermediate_auction_threshold:
                    solution = self.run_intermediate_auction(instance, solution)
                    try:
                        intermediate_auction_threshold = next(intermediate_auction_times)
                    except StopIteration:
                        intermediate_auction_threshold = dt.datetime.max  # or better, = ut.END_TIME?

            assert request in solution.unassigned_requests
            carrier_id = instance.request_to_carrier_assignment[request]
            carrier = solution.carriers[carrier_id]
            solution.assign_requests_to_carriers([request], [carrier_id])

            # ===== Request Acceptance and Time Window Management =====
            self.request_acceptance_and_time_window(instance, solution, carrier, request)

        # ===== Final Improvement =====
        # plot_vienna_vrp_solution(instance, solution)  # REMOVEME for debugging only
        before_improvement = solution.objective()
        timer = pr.Timer()
        solution = self.tour_improvement.execute(instance, solution)
        timer.write_duration_to_solution(solution, 'runtime_final_improvement')
        if isinstance(before_improvement, dt.timedelta):
            assert int(before_improvement.total_seconds()) >= int(solution.objective().total_seconds()), instance.id_
        else:
            assert int(before_improvement) >= int(solution.objective()), instance.id_
        ut.validate_solution(instance, solution)  # safety check to make sure everything's functional

        # ===== Final Auction =====
        if self.final_auction:
            timer = pr.Timer()
            solution = self.final_auction.execute(instance, solution)
            timer.write_duration_to_solution(solution, 'runtime_final_auction')

        ut.validate_solution(instance, solution)  # safety check to make sure everything's functional
        logger.log(SUCCESS, f'{instance.id_}: Success {solution.solver_config}')

        # plot_vienna_vrp_solution(instance, solution)  # REMOVEME for debugging only
        return instance, solution

    def request_acceptance_and_time_window(self, instance: it.MDVRPTWInstance, solution: slt.CAHDSolution, carrier,
                                           request: int):
        delivery_vertex = instance.vertex_from_request(request)
        acceptance_type, selected_tw = self.request_acceptance.execute(instance, carrier, request)

        if acceptance_type == 'accept_feasible':
            carrier.accepted_requests.append(request)
            instance.assign_time_window(delivery_vertex, selected_tw)
            self.tour_construction.insert_single_request(instance, solution, carrier.id_, request)

        elif acceptance_type == 'accept_infeasible':
            logger.debug(f'[{instance.id_}] No feasible TW can be offered from Carrier {carrier.id_} '
                         f'to request {request}: {acceptance_type}')
            carrier.accepted_infeasible_requests.append(request)
            instance.assign_time_window(delivery_vertex, selected_tw)
            self.tour_construction.create_pendulum_tour_for_infeasible_request(instance, solution, carrier.id_, request)

        else:
            logger.debug(f'[{instance.id_}] No feasible TW can be offered from Carrier {carrier.id_} '
                         f'to request {request}: {acceptance_type}')
            instance.assign_time_window(delivery_vertex, ut.EXECUTION_TIME_HORIZON)
            carrier.rejected_requests.append(request)
            carrier.unrouted_requests.remove(request)

        # update acceptance rate
        carrier.acceptance_rate = len(
            carrier.accepted_requests + carrier.accepted_infeasible_requests) / len(
            carrier.assigned_requests)

    '''
    def request_acceptance_and_time_window(self, instance, solution, carrier, request):
        pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(request)
        instance.assign_time_window(pickup_vertex, ut.EXECUTION_TIME_HORIZON)  # pickup at any time
        acceptance_type, selected_tw = self.request_acceptance.execute(instance, carrier, request)

        if acceptance_type == 'accept_feasible':
            carrier.accepted_requests.append(request)
            instance.assign_time_window(delivery_vertex, selected_tw)
            self.tour_construction.insert_single_request(instance, solution, carrier.id_, request)

        elif acceptance_type == 'accept_infeasible':
            logger.debug(f'[{instance.id_}] No feasible TW can be offered from Carrier {carrier.id_} '
                         f'to request {request}: {acceptance_type}')
            carrier.accepted_infeasible_requests.append(request)
            instance.assign_time_window(delivery_vertex, selected_tw)
            self.tour_construction.create_pendulum_tour_for_infeasible_request(instance, solution, carrier.id_, request)

        else:
            logger.debug(f'[{instance.id_}] No feasible TW can be offered from Carrier {carrier.id_} '
                         f'to request {request}: {acceptance_type}')
            instance.assign_time_window(delivery_vertex, ut.EXECUTION_TIME_HORIZON)
            carrier.rejected_requests.append(request)
            carrier.unrouted_requests.remove(request)

        # update acceptance rate
        carrier.acceptance_rate = len(
            carrier.accepted_requests + carrier.accepted_infeasible_requests) / len(
            carrier.assigned_requests)
    '''

    def run_intermediate_auction(self, instance, solution):
        timer = pr.Timer()
        # solution = self.tour_improvement.execute(instance, solution)
        timer.write_duration_to_solution(solution, 'runtime_intermediate_improvements', True)
        timer = pr.Timer()
        solution = self.intermediate_auction.execute(instance, solution)
        timer.write_duration_to_solution(solution, 'runtime_intermediate_auctions', True)
        return solution
