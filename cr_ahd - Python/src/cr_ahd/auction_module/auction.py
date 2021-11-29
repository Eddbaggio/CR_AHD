import logging
import warnings
from abc import ABC
from copy import deepcopy
from typing import List

import pandas as pd

from auction_module import request_selection as rs, bundle_generation as bg, bidding as bd, \
    winner_determination as wd
from core_module import instance as it, solution as slt
from routing_module import tour_construction as cns, metaheuristics as mh
from utility_module import profiling as pr, utils as ut

logger = logging.getLogger(__name__)


class Auction:
    def __init__(self,
                 tour_construction: cns.PDPParallelInsertionConstruction,
                 tour_improvement: mh.PDPTWMetaHeuristic,
                 request_selection: rs.RequestSelectionBehavior,
                 bundle_generation: bg.LimitedBundlePoolGenerationBehavior,
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

    def execute(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution):
        for auction_round in range(self.num_auction_rounds):

            # auction
            pre_auction_solution = deepcopy(solution)
            solution, winners = self.reallocate_requests(instance, solution)

            # clears the solution and re-optimize
            solution = self.re_optimize(instance, solution, winners)

            # consistency check. cannot compare carriers' profit individually before and after the auction because I
            # do not implement profit sharing! Thus, an individual carrier may be worse off, while the global solution
            # is better!
            if not pre_auction_solution.objective() <= solution.objective():
                """
                Unfortunately, there are several circumstances that can lead to the post-auction result being worse
                than the pre-auction solution. In that case, throw a warning and recover the pre-auction solution!
                Causes:
                
                (a) If intermediate auctions are used, any auction other than the first may run into the following 
                problem: The sorting of assigned requests for a given carrier can be in no order due to request 
                reassignments in a previous auction. Now, if the current auction's bidding takes place and the carrier
                bids on his original bundle, this bundle will be in a different order, meaning that dynamic insertion
                obtains a different - potentially worse - result! SOLUTION: make sure that the dynamic insertion
                (of the bidding phase) follows the same order as the carrier.assigned_requests list.
                -> FIXED by adding request_disclosure_time to the instance
                
                (b) the metaheuristic used for improvements does not work correctly and returns worse results. This
                messes up the bidding because the bid on a given carrier's original bundle may be incorrect.
                -> possible, but let's first assume this is not the real issue
                
                (c) the bidding process inserts all requests from scratch and runs a single improvement at the end. 
                with intermediate auctions, there is an improvement phase after each intermediate auction and only 
                afterwards will newly arriving requests be inserted. Thus, the bidding (on one's own requests) is not 
                the same as the dynamic insertion because the latter has additional intermediate improvements. 
                -> is not the (only) problem at the moment since using NoMetaheuristic did not resolve the issue 
                
                (d) after submitting requests to the auction pool, the solution is not re-optimized from scratch but
                left as is. On the other hand, post_auction_routing re-optimizes from scratch. The re-optimization
                may yield results that are worse than simply removing requests!
                -> FIXED by only re-optimizing the winners' route plan and leaving carriers untouched that did not win
                a bundle. However, this does not fix issue (e)! 
                
                (e) similarly to (d) the following procedure causes problems, too: A carrier submits something in the 
                first auction but does not win anything in this auction. thus, his route plan will never be 
                re-optimized after request selection. He is left with the solution [A] obtained by simply removing 
                the submitted requests from their tour. Dynamic insertion of any request that arrives after the first 
                auction is now based on [A]. In a second auction, the same carrier may win *his own* submitted 
                requests. Since he is technically a winner, his route plan will be re-optimized *from scratch*, 
                potentially making different insertion decisions as were made when insertion was based on [A]. The 
                re-optimization may yield results that are worse than solution [A] (or even infeasible)! 
                -> TODO Fix this by doing a re-optimization after RS for all carriers?? 
                    -> No, because this may be infeasible, too
                
                """

                raise ValueError(f'{instance.id_},:\n'
                                 f' {solution.objective()} < {pre_auction_solution.objective()}\n'
                                 f' Post-auction objective is lower than pre-auction objective!,'
                                 f' Recovering the pre-auction solution')
                solution = pre_auction_solution
                assert pre_auction_solution.objective() == solution.objective()
        return solution

    def reallocate_requests(self,
                            instance: it.MDPDPTWInstance,
                            solution: slt.CAHDSolution) -> (slt.CAHDSolution, List[int]):
        logger.debug(f'running auction {self.__class__.__name__}')

        pre_rs_solution = deepcopy(solution)

        # ===== [1] Request Selection =====
        timer = pr.Timer()
        auction_request_pool, original_bundling_labels = self.request_selection.execute(instance, solution)

        # /// Part of the sequential-auctions problem
        # post_rs_solution must be re-optimized to avoid inconsistencies when intermediate auctions are used
        # however, a re-optimization is not guaranteed to be feasible unless exact approach is used
        # solution = self.re_optimize(instance, solution)
        # ///

        original_bundles = ut.indices_to_nested_lists(original_bundling_labels, auction_request_pool)
        # timer.write_duration_to_solution(solution, 'runtime_request_selection') fixme need to distinguish between intermediate and final: auction_counter in Solver class?

        if auction_request_pool:
            profit_after_rs = [carrier.sum_profit() for carrier in solution.carriers]
            logger.debug(f'requests {auction_request_pool} have been submitted to the auction pool')

            # ===== [2] Bundle Generation =====
            timer = pr.Timer()
            auction_bundle_pool = self.bundle_generation.execute(instance, solution, auction_request_pool,
                                                                 original_bundling_labels)
            # timer.write_duration_to_solution(solution, 'runtime_auction_bundle_pool_generation') fixme
            logger.debug(f'bundles {auction_bundle_pool} have been created')

            # ===== [3] Bidding =====
            logger.debug(f'Generating bids_matrix')
            timer = pr.Timer()
            bids_matrix = self.bidding.execute_bidding(instance, solution, auction_bundle_pool)

            # /// REMOVEME only for debugging
            bids_dict = {tuple(bundle): bids for bundle, bids in zip(auction_bundle_pool, bids_matrix)}
            bids_on_original_bundles = []
            for bundle, carrier in zip(original_bundles, solution.carriers):
                bids_on_original_bundles.append(bids_dict[tuple(bundle)][carrier.id_])
            # ///

            # timer.write_duration_to_solution(solution, 'runtime_bidding') fixme
            logger.debug(f'Bids {bids_matrix} have been created for bundles {auction_bundle_pool}')

            # ===== [4.1] Winner Determination =====
            timer = pr.Timer()
            winner_bundles, bundle_winners = self.winner_determination.execute(instance, solution,
                                                                               auction_request_pool,
                                                                               auction_bundle_pool, bids_matrix)
            # timer.write_duration_to_solution(solution, 'runtime_winner_determination') fixme

            # /// REMOVEME only for debugging
            winning_bids = []
            for winner, bundle in zip(bundle_winners, winner_bundles):
                winning_bids.append(bids_dict[tuple(bundle)][winner])
            # ///

            # todo: store whether the auction did achieve a reallocation or not

            # ===== [4.2] Bundle Reallocation =====
            self.assign_bundles_to_winners(instance, solution, winner_bundles, bundle_winners)

        else:
            logger.warning(f'No requests have been submitted!')

        return solution, bundle_winners

    def re_optimize(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution, carrier_ids: List[int] = None):
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
            while len(carrier.unrouted_requests) > 0:
                request = carrier.unrouted_requests[0]
                self.tour_construction.insert_single_request(instance, solution, carrier.id_, request)
        # timer.write_duration_to_solution(solution, 'runtime_final_dynamic_insertion') FixMe
        timer = pr.Timer()
        solution = self.tour_improvement.execute(instance, solution)
        # timer.write_duration_to_solution(solution, 'runtime_final_improvement') fixme
        return solution

    @staticmethod
    def assign_bundles_to_winners(instance: it.MDPDPTWInstance, solution: slt.CAHDSolution,
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


# ======================================================================================================================
# THESE CLASSES ARE NOT YET IN USE
# ======================================================================================================================

class Bundle(list):
    """bundle: Sequence[int]
    a sequence of request indices that make up one bundle -> cannot have duplicates etc., maybe a set rather than a list
    would be better?"""
    pass


class Bundling(list):
    """bundling: Sequence[Sequence[int]]
    a sequence of {bundles} (see above) that fully partition the {auction_request_pool}"""
    pass


class BundlingLabels(list):
    """bundling_labels: Sequence[int]
    a sequence of bundle indices that partitions the {auction_request_pool}
     NOTE: Contrary to the {bundling}, the {bundling_labels} is not nested and does not contain request indices but
     bundle indices"""
    pass


class AuctionRequestPool(list):
    """auction_request_pool: Sequence[int][int]
    a sequence of request indices of the requests that were submitted to be auctioned"""
    pass


class AuctionBundlePool(list):
    """auction_bundle_pool: Sequence[Sequence[int]]
    a nested sequence of request index sequences. Each inner sequence is a {bundle} inside the auction_bundle_pool that
    carriers will have to bid on"""
    pass
