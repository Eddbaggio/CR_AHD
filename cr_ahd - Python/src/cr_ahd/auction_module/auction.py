import logging
from abc import ABC

from src.cr_ahd.auction_module import request_selection as rs, bundle_generation as bg, bidding as bd, \
    winner_determination as wd
from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import tour_construction as cns, metaheuristics as mh
from src.cr_ahd.utility_module import utils as ut

logger = logging.getLogger(__name__)


class Auction(ABC):
    def __init__(self,
                 tour_construction: cns.PDPParallelInsertionConstruction,
                 tour_improvement: mh.PDPMetaHeuristic,
                 request_selection: rs.RequestSelectionBehavior,
                 bundle_generation: bg.BundleSetGenerationBehavior,
                 bidding: bd.BiddingBehavior,
                 winner_determination: wd.WinnerDeterminationBehavior,
                 reopt_and_improve_after_request_selection: bool,
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
        self.reopt_and_improve_after_request_selection = reopt_and_improve_after_request_selection

        assert isinstance(self.bidding.tour_construction, type(self.tour_construction))
        assert isinstance(self.bidding.tour_improvement, type(self.tour_improvement))

    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        logger.debug(f'running auction {self.__class__.__name__}')
        # Request Selection
        auction_pool_requests, original_bundles_indices = self.request_selection.execute(instance, solution,
                                                                                         ut.NUM_REQUESTS_TO_SUBMIT)
        if auction_pool_requests:
            logger.debug(f'requests {auction_pool_requests} have been submitted to the auction pool')

            # optional reoptimization, not all auction variants to a reopt, the abstract method may be empty
            if self.reopt_and_improve_after_request_selection:
                self._reopt_and_improve(instance, solution)

            # Bundle Generation
            # TODO maybe bundles should be a list of bundle indices rather than a list of lists of request indices?
            auction_pool_bundles = self.bundle_generation.execute(instance, solution, auction_pool_requests,
                                                                  original_bundles_indices)
            original_bundles = ut.indices_to_nested_lists(original_bundles_indices, auction_pool_requests)
            original_bundles_indices = [auction_pool_bundles.index(x) for x in original_bundles]
            logger.debug(f'bundles {auction_pool_bundles} have been created')

            # Bidding
            logger.debug(f'Generating bids_matrix')
            bids_matrix = self.bidding.execute(instance, solution, auction_pool_bundles)
            logger.debug(f'Bids {bids_matrix} have been created for bundles {auction_pool_bundles}')

            # Winner Determination
            winner_bundles, bundle_winners = self.winner_determination.execute(instance, solution,
                                                                               auction_pool_requests,
                                                                               auction_pool_bundles, bids_matrix)
            winner_bundles_indices = [auction_pool_bundles.index(x) for x in winner_bundles]

            # bundle reallocation
            self.assign_bundles_to_winners(solution, winner_bundles, bundle_winners)

        else:
            logger.warning(f'No requests have been submitted!')

        # final routing
        self._reopt_and_improve(instance, solution)

        solution.solver_config['auction_tour_construction'] = self.tour_construction.__class__.__name__
        solution.solver_config['auction_tour_improvement'] = self.tour_improvement.__class__.__name__
        solution.solver_config['request_selection'] = self.request_selection.__class__.__name__
        solution.solver_config[
            'reopt_and_improve_after_request_selection'] = self.reopt_and_improve_after_request_selection
        solution.solver_config['bundle_generation'] = self.bundle_generation.__class__.__name__
        solution.solver_config['bidding'] = self.bidding.__class__.__name__
        solution.solver_config['winner_determination'] = self.winner_determination.__class__.__name__
        pass

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

    def _reopt_and_improve(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        # clear the solution and do a dynamic re-optimization including improvement
        solution.clear_carrier_routes()
        for carrier in range(len(solution.carriers)):
            while solution.carriers[carrier].unrouted_requests:
                request = solution.carriers[carrier].unrouted_requests[0]
                self.tour_construction.construct_dynamic(instance, solution, carrier)
        self.tour_improvement.execute(instance, solution)
        pass
