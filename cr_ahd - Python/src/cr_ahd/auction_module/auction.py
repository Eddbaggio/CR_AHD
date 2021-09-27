import logging
import warnings
from abc import ABC
from copy import deepcopy

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
            pre_auction_objective = solution.objective()
            # print([(carrier.routed_requests, carrier.unrouted_requests) for carrier in solution.carriers])  # RemoveMe
            solution_backup = deepcopy(solution)
            solution = self.reallocate_requests(instance, solution)
            # print([(carrier.routed_requests, carrier.unrouted_requests) for carrier in solution.carriers])  # RemoveMe

            # clears the solution and re-optimize
            solution = self.post_auction_routing_ClearAndReinsertAll(instance, solution)

            if not pre_auction_objective <= solution.objective():
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
                (b) the metaheuristic used for improvements does not work correctly and returns worse results. This
                messes up the bidding because the bid on a given carrier's original bundle may be incorrect
                """
                raise ValueError(f'{instance.id_}: Post-auction objective is lower than pre-auction objective!,'
                                 f' Recovering the pre-auction solution')
                solution = solution_backup
                assert pre_auction_objective == solution.objective()
        return solution

    def reallocate_requests(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution) -> slt.CAHDSolution:
        logger.debug(f'running auction {self.__class__.__name__}')

        # ===== [1] Request Selection =====
        timer = pr.Timer()
        auction_request_pool, original_bundling_labels = self.request_selection.execute(instance, solution)
        original_bundles = ut.indices_to_nested_lists(original_bundling_labels, auction_request_pool)
        # timer.write_duration_to_solution(solution, 'runtime_request_selection') fixme

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
            # timer.write_duration_to_solution(solution, 'runtime_bidding') fixme
            logger.debug(f'Bids {bids_matrix} have been created for bundles {auction_bundle_pool}')

            # ===== [4.1] Winner Determination =====
            timer = pr.Timer()
            winner_bundles, bundle_winners = self.winner_determination.execute(instance, solution,
                                                                               auction_request_pool,
                                                                               auction_bundle_pool, bids_matrix)
            # timer.write_duration_to_solution(solution, 'runtime_winner_determination') fixme
            # todo: store whether the auction did achieve a reallocation or not

            # ===== [4.2] Bundle Reallocation =====
            self.assign_bundles_to_winners(solution, winner_bundles, bundle_winners)

        else:
            logger.warning(f'No requests have been submitted!')

        return solution

    def post_auction_routing_ClearAndReinsertAll(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution):
        """
        After requests have been reallocated, this function builds the new tours based on the new requests. It is
        important that this routing follows the same steps as the routing approach in the bidding phase to reliably
        obtain feasible and consistent solutions.
        Currently, this function performs a complete optimization from scratch: dynamic insertion + improvement
        :param instance:
        :param solution:
        :return:
        """
        solution.clear_carrier_routes(None)  # clear everything that was left after Request Selection
        timer = pr.Timer()
        for carrier in solution.carriers:
            while carrier.unrouted_requests:
                request = carrier.unrouted_requests[0]
                self.tour_construction.insert_single_request(instance, solution, carrier.id_, request)
        # timer.write_duration_to_solution(solution, 'runtime_final_dynamic_insertion') FixMe
        timer = pr.Timer()
        solution = self.tour_improvement.execute(instance, solution)
        # timer.write_duration_to_solution(solution, 'runtime_final_improvement') fixme
        return solution

    def post_auction_routing_InsertBundle(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution):
        """
        Does NOT clear the routes before inserting yet unrouted requests
        :param instance:
        :param solution:
        :return:
        """
        raise NotImplementedError(f'This has never been used before, check before removing the Error raise')
        for carrier in solution.carriers:
            while carrier.unrouted_requests:
                request = carrier.unrouted_requests[0]
                self.tour_construction.insert_single_request(instance, solution, carrier.id_, request)
        solution = self.tour_improvement.execute(instance, solution)
        return solution

    @staticmethod
    def assign_bundles_to_winners(solution, winner_bundles, bundle_winners):
        # assign the bundles to the corresponding winner
        for bundle, winner in zip(winner_bundles, bundle_winners):
            solution.assign_requests_to_carriers(bundle, [winner] * len(bundle))
            solution.carriers[winner].accepted_requests.extend(bundle)
        # must be sorted to obtain the acceptance phase's solutions also in the final routing
        for carrier_ in solution.carriers:
            carrier_.assigned_requests.sort()
            carrier_.accepted_requests.sort()
            carrier_.unrouted_requests.sort()


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
