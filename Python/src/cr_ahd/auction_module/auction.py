import logging
from copy import deepcopy
from typing import List
import pandas as pd
from gurobipy import GRB

from auction_module import request_selection as rs, bidding as bd, winner_determination as wd
from auction_module.bundle_generation import bundle_gen as bg, partition_based_bg as bgp
from core_module import instance as it, solution as slt
from routing_module import tour_construction as cns, metaheuristics as mh
from utility_module import profiling as pr, utils as ut
from utility_module.io import output_dir, unique_path

logger = logging.getLogger(__name__)


class Auction:
    def __init__(self,
                 tour_construction: cns.VRPTWInsertionConstruction,
                 tour_improvement: mh.VRPTWMetaHeuristic,
                 request_selection: rs.RequestSelectionBehavior,
                 bundle_generation: bg.BundleGenerationBehavior,
                 bidding: bd.BiddingBehavior,
                 winner_determination: wd.WinnerDeterminationBehavior,
                 num_auction_rounds: int = 1
                 ):
        """
        Auction class can be called with various parameters to create different auction variations

        :param tour_construction: construction method
        :param tour_improvement: improvement metaheuristic
        :param request_selection: request selection method
        :param bundle_generation:
        :param bidding: bidding behavior, in bidding, the same construction & improvement methods must be used as
        are specified for this Auction to guarantee consistent, individual-rational auction results
        :param winner_determination:
        """

        self.tour_construction = tour_construction
        self.tour_improvement = tour_improvement
        self.request_selection = request_selection
        self.bundle_generation = bundle_generation
        self.bidding = bidding
        self.winner_determination = winner_determination
        self.num_auction_rounds = num_auction_rounds

        assert isinstance(self.bidding.tour_construction, type(self.tour_construction))
        assert isinstance(self.bidding.tour_improvement, type(self.tour_improvement))

    def execute(self, instance: it.CAHDInstance, solution: slt.CAHDSolution):
        for auction_round in range(self.num_auction_rounds):

            # auction
            pre_auction_solution = deepcopy(solution)
            status, solution, winners = self.reallocate_requests(instance, solution)

            if status == GRB.OPTIMAL:
                # clears the solution and re-optimize
                solution = self.re_optimize(instance, solution, winners)

                # consistency check
                if solution.objective() > pre_auction_solution.objective():
                    raise ValueError(f'{instance.id_},:\n'
                                     f' Post={solution.objective()}; Pre={pre_auction_solution.objective()}\n'
                                     f' Post-auction objective is worse than pre-auction objective!\n'
                                     f' Recovering the pre-auction solution.')
                    solution = pre_auction_solution
                    assert pre_auction_solution.objective() == solution.objective()
        return solution

    def reallocate_requests(self,
                            instance: it.CAHDInstance,
                            solution: slt.CAHDSolution) -> (slt.CAHDSolution, List[int]):
        logger.debug(f'running auction {self.__class__.__name__}')

        pre_rs_solution = deepcopy(solution)

        # ===== [1] Request Selection =====
        timer = pr.Timer()
        auction_request_pool, original_partition_labels = self.request_selection.execute(instance, solution)

        original_bundles = ut.indices_to_nested_lists(original_partition_labels, auction_request_pool)
        timer.write_duration_to_solution(solution, 'runtime_request_selection')

        if auction_request_pool:
            logger.debug(f'requests {auction_request_pool} have been submitted to the auction pool')

            # ===== [2] Bundle Generation =====
            timer = pr.Timer()
            auction_bundle_pool = self.bundle_generation.execute(instance, solution, auction_request_pool,
                                                                 original_partition_labels)
            timer.write_duration_to_solution(solution, 'runtime_auction_bundle_pool_generation')
            logger.debug(f'bundles {auction_bundle_pool} have been created')

            # ===== [3] Bidding =====
            logger.debug(f'Generating bids_matrix')
            timer = pr.Timer()
            bids_matrix = self.bidding.execute_bidding(instance, solution, auction_bundle_pool)

            # /// write bids matrix to data/output/bids
            path = output_dir.joinpath('bids/')
            path.mkdir(exist_ok=True, parents=True)
            path = unique_path(path, f'bids_{instance.id_}' + '_#{:03d}' + '.csv')
            bids_df = pd.DataFrame(data=((x.total_seconds() for x in y) for y in bids_matrix),
                                   index=['/'.join(str(x) for x in y) for y in auction_bundle_pool])
            bids_df.to_csv(path)
            # ///

            timer.write_duration_to_solution(solution, 'runtime_bidding')
            logger.debug(f'Bids {bids_matrix} have been created for bundles {auction_bundle_pool}')

            # ===== [4.1] Winner Determination =====
            timer = pr.Timer()
            wdp_solution = self.winner_determination.execute(
                instance, solution, auction_request_pool, auction_bundle_pool, bids_matrix, original_partition_labels)
            status, winner_bundles, bundle_winners, winner_bids, winner_partition_labels = wdp_solution

            hamming_dist = ut.hamming_distance(original_partition_labels, winner_partition_labels)
            degree_of_reallocation = hamming_dist / len(original_partition_labels)
            solution.degree_of_reallocation = degree_of_reallocation

            timer.write_duration_to_solution(solution, 'runtime_winner_determination')
            if status != GRB.OPTIMAL or -GRB.INFINITY in winner_bids:  # fall back to the pre-auction solution
                solution = pre_rs_solution

            else:
                # ===== [4.2] Bundle Reallocation =====
                self.assign_bundles_to_winners(instance, solution, winner_bundles, bundle_winners)

        else:
            logger.warning(f'No requests have been submitted!')

        return status, solution, bundle_winners

    def re_optimize(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, carrier_ids: List[int] = None):
        """
        After requests have been reallocated, this function builds the new tours based on the new requests. It is
        important that this routing follows the same steps as the routing approach in the bidding phase to reliably
        obtain feasible and consistent solutions.
        Currently, this function performs a complete optimization from scratch: dynamic insertion + improvement
        :param instance:
        :param solution:
        :return:
        """
        if carrier_ids is None:
            carrier_ids = [carrier.id_ for carrier in solution.carriers]

        solution.clear_carrier_routes(carrier_ids)  # for all winners, clear what was left after Request Selection
        timer = pr.Timer()
        for carrier_id in carrier_ids:
            carrier = solution.carriers[carrier_id]
            for request in carrier.accepted_infeasible_requests:
                self.tour_construction.create_pendulum_tour_for_infeasible_request(instance, solution, carrier_id,
                                                                                   request)
            while len(carrier.unrouted_requests) > 0:
                request = carrier.unrouted_requests[0]
                self.tour_construction.insert_single_request(instance, solution, carrier.id_, request)
        timer.write_duration_to_solution(solution, 'runtime_reopt_dynamic_insertion')
        timer = pr.Timer()
        solution = self.tour_improvement.execute(instance, solution)
        timer.write_duration_to_solution(solution, 'runtime_reopt_improvement')
        return solution

    @staticmethod
    def assign_bundles_to_winners(instance: it.CAHDInstance, solution: slt.CAHDSolution,
                                  winner_bundles: List[List[int]],
                                  bundle_winners: List[int]):
        # assign the bundles to the corresponding winner
        for bundle, winner in zip(winner_bundles, bundle_winners):
            solution.assign_requests_to_carriers(bundle, [winner] * len(bundle))
            solution.carriers[winner].accepted_requests.extend(bundle)
        # must be sorted to obtain the acceptance phase's solutions also in the final routing
        for carrier in solution.carriers:
            carrier.assigned_requests.sort(
                key=lambda x: (instance.request_disclosure_time[x], instance.request_to_carrier_assignment[x]))
            carrier.accepted_requests.sort(
                key=lambda x: (instance.request_disclosure_time[x], instance.request_to_carrier_assignment[x]))
            carrier.unrouted_requests.sort(
                key=lambda x: (instance.request_disclosure_time[x], instance.request_to_carrier_assignment[x]))
