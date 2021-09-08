import logging
from abc import ABC
from copy import deepcopy

from src.cr_ahd.auction_module import request_selection as rs, bundle_generation as bg, bidding as bd, \
    winner_determination as wd
from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import tour_construction as cns, metaheuristics as mh
from src.cr_ahd.utility_module import utils as ut, profiling as pr

logger = logging.getLogger(__name__)


class Auction(ABC):
    def __init__(self,
                 tour_construction: cns.PDPParallelInsertionConstruction,
                 tour_improvement: mh.PDPTWMetaHeuristic,
                 request_selection: rs.RequestSelectionBehavior,
                 bundle_generation: bg.LimitedBundlePoolGenerationBehavior,
                 bidding: bd.BiddingBehavior,
                 winner_determination: wd.WinnerDeterminationBehavior,
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
        :param reopt_and_improve_after_request_selection: shall a reoptimization happen after the requests have been
        submitted? As long as I'm doing complete, dynamic reoptimization in bidding, this should be set to True
        """

        self.tour_construction = tour_construction
        self.tour_improvement = tour_improvement
        self.request_selection = request_selection
        self.bundle_generation = bundle_generation
        self.bidding = bidding
        self.winner_determination = winner_determination

        assert isinstance(self.bidding.tour_construction, type(self.tour_construction))
        assert isinstance(self.bidding.tour_improvement, type(self.tour_improvement))

    def execute_auction(self, instance: it.MDPDPTWInstance, solution: slt.CAHDSolution) -> slt.CAHDSolution:
        logger.debug(f'running auction {self.__class__.__name__}')

        # ===== Request Selection =====
        timer = pr.Timer()
        auction_request_pool, original_bundling_labels = self.request_selection.execute(instance, solution)
        timer.write_duration_to_solution(solution, 'runtime_request_selection')

        if auction_request_pool:
            logger.debug(f'requests {auction_request_pool} have been submitted to the auction pool')

            # ===== Bundle Generation =====
            timer = pr.Timer()
            auction_bundle_pool = self.bundle_generation.execute(instance, solution, auction_request_pool,
                                                                 original_bundling_labels)
            timer.write_duration_to_solution(solution, 'runtime_auction_bundle_pool_generation')
            logger.debug(f'bundles {auction_bundle_pool} have been created')

            # ===== Bidding =====
            logger.debug(f'Generating bids_matrix')
            timer = pr.Timer()
            bids_matrix = self.bidding.execute_bidding(instance, solution, auction_bundle_pool)
            timer.write_duration_to_solution(solution, 'runtime_bidding')
            logger.debug(f'Bids {bids_matrix} have been created for bundles {auction_bundle_pool}')

            # ===== Winner Determination =====
            timer = pr.Timer()
            winner_bundles, bundle_winners = self.winner_determination.execute(instance, solution,
                                                                               auction_request_pool,
                                                                               auction_bundle_pool, bids_matrix)
            timer.write_duration_to_solution(solution, 'runtime_winner_determination')
            # todo: store whether the auction did achieve a reallocation or not

            # ===== Bundle Reallocation =====
            self.assign_bundles_to_winners(solution, winner_bundles, bundle_winners)

        else:
            logger.warning(f'No requests have been submitted!')

        # ===== Final Routing =====
        # clear the solution and do a dynamic re-optimization + improvement
        solution.clear_carrier_routes()
        timer = pr.Timer()
        for carrier in solution.carriers:
            while carrier.unrouted_requests:
                request = carrier.unrouted_requests[0]
                self.tour_construction.insert_single(instance, solution, carrier.id_, request)
        timer.write_duration_to_solution(solution, 'runtime_final_construction')
        timer = pr.Timer()
        solution = self.tour_improvement.execute(instance, solution)
        timer.write_duration_to_solution(solution, 'runtime_final_improvement')

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
