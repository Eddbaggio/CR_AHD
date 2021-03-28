import abc
import logging.config
from typing import final

import src.cr_ahd.auction_module.auction as au
import src.cr_ahd.auction_module.bidding as bd
import src.cr_ahd.auction_module.bundle_generation as bg
import src.cr_ahd.auction_module.request_selection as rs
import src.cr_ahd.auction_module.winner_determination as wd
import src.cr_ahd.utility_module.utils as ut
from src.cr_ahd.core_module import instance as it
from src.cr_ahd.solving_module.tour_construction import SequentialCheapestInsertion, I1Insertion
from src.cr_ahd.solving_module.tour_improvement import TwoOpt
from src.cr_ahd.solving_module.tour_initialization import EarliestDueDate

logger = logging.getLogger(__name__)


class Solver(abc.ABC):
    def execute(self, instance):
        """
        apply the concrete solution algorithm
        :param instance:
        :return: None
        """
        self._solve(instance)
        instance.solution_algorithm = self
        pass

    @abc.abstractmethod
    def _solve(self, instance: it.Instance):
        pass


class StaticSequentialInsertion(Solver):
    """
    Initializes pendulum tours first! Then builds the routes. The Dynamic Version does NOT initialize tours
    """

    def _solve(self, instance: it.Instance):
        self.assign_all_requests(instance)
        self.initialize_pendulum_tours(instance)
        self.build_routes(instance)
        self.finalize_with_local_search(instance)
        pass

    def assign_all_requests(self, instance):
        for request in instance.unrouted_requests:
            carrier = ut.get_carrier_by_id(instance.carriers, request.carrier_assignment)
            carrier.assign_requests([request])

    def initialize_pendulum_tours(self, instance):
        EarliestDueDate().initialize_instance(instance)  # not flexible!
        pass

    def build_routes(self, instance):
        SequentialCheapestInsertion().solve_instance(instance)
        pass

    def finalize_with_local_search(self, instance):
        TwoOpt().improve_instance(instance)
        pass


class StaticI1Insertion(Solver):
    def _solve(self, instance: it.Instance):
        self.assign_all_requests(instance)
        self.initialize_pendulum_tours(instance)
        carrier = True
        while carrier:
            carrier = self.build_routes(instance)
            if carrier:
                self.initialize_another_tour(carrier)
        self.finalize_with_local_search(instance)
        pass

    def assign_all_requests(self, instance):
        for request in instance.unrouted_requests:
            carrier = ut.get_carrier_by_id(instance.carriers, request.carrier_assignment)
            carrier.assign_requests([request])

    def initialize_pendulum_tours(self, instance):
        EarliestDueDate().initialize_instance(instance)  # not flexible!
        pass

    def initialize_another_tour(self, carrier):
        EarliestDueDate().initialize_carrier(carrier)

    def build_routes(self, instance):
        carrier = I1Insertion().solve_instance(instance)
        return carrier

    def finalize_with_local_search(self, instance):
        TwoOpt().improve_instance(instance)
        pass


class StaticI1InsertionWithAuction(Solver):
    def _solve(self, instance: it.Instance):
        self.assign_all_requests(instance)
        if instance.num_carriers > 1:
            self.run_auction(instance)
        carrier = True
        while carrier:
            carrier = self.build_routes(instance)
            if carrier:
                self.initialize_another_tour(carrier)
        self.finalize_with_local_search(instance)

    def assign_all_requests(self, instance):
        for request in instance.unrouted_requests:
            carrier = ut.get_carrier_by_id(instance.carriers, request.carrier_assignment)
            carrier.assign_requests([request])

    def run_auction(self, instance):
        submitted_requests = rs.HighestInsertionCostDistance().execute(instance, 0.5)
        bundle_set = bg.RandomPartition(instance.distance_matrix).execute(submitted_requests)
        bids = bd.I1TravelDistanceIncrease().execute(bundle_set, instance.carriers)
        wd.LowestBid().execute(bids)

    def initialize_another_tour(self, carrier):
        EarliestDueDate().initialize_carrier(carrier)

    def build_routes(self, instance):
        carrier = I1Insertion().solve_instance(instance)
        return carrier

    def finalize_with_local_search(self, instance):
        TwoOpt().improve_instance(instance)


# =====================================================================================================================
# DYNAMIC
# =====================================================================================================================


class DynamicSolver(Solver):
    @final
    def _solve(self, instance: it.Instance):
        while instance.unrouted_requests:
            self.assign_n_requests(instance, ut.opts['dynamic_cycle_time'])
            self._solve_cycle(instance)
        self.finalize_with_local_search(instance)

    @final
    def assign_n_requests(self, instance, n):
        logger.debug(f'Assigning {n} requests to carriers')
        for request in instance.unassigned_requests()[:n]:
            carrier = ut.get_carrier_by_id(instance.carriers, request.carrier_assignment)
            carrier.assign_requests([request])

    @final
    def finalize_with_local_search(self, instance):
        TwoOpt().improve_instance(instance)
        pass

    @abc.abstractmethod
    def _solve_cycle(self, instance):
        pass


class DynamicSequentialInsertion(DynamicSolver):
    """
    Does NOT initialize pendulum tours!
    """

    def _solve_cycle(self, instance):
        SequentialCheapestInsertion().solve_instance(instance)
        pass


class DynamicI1Insertion(DynamicSolver):
    """
    ADD DOCSTRING!!
    """

    def _solve_cycle(self, instance):
        carrier = True
        while carrier:
            carrier = self.build_routes(instance)
            if carrier:
                self.initialize_another_tour(carrier)

    def initialize_another_tour(self, carrier):
        EarliestDueDate().initialize_carrier(carrier)

    def build_routes(self, instance):
        carrier = I1Insertion().solve_instance(instance)
        return carrier


class DynamicI1InsertionWithAuctionA(DynamicSolver):
    """
    ADD DOCSTRING!!
    """

    def _solve_cycle(self, instance):
        if instance.num_carriers > 1:
            self.run_auction(instance)
        carrier = True
        while carrier:
            carrier = self.build_routes(instance)
            if carrier:
                self.initialize_another_tour(carrier)

    def run_auction(self, instance):
        au.Auction_a().execute(instance)

    def initialize_another_tour(self, carrier):
        EarliestDueDate().initialize_carrier(carrier)

    def build_routes(self, instance):
        carrier = I1Insertion().solve_instance(instance)
        return carrier


class DynamicI1InsertionWithAuctionB(DynamicSolver):
    """
    ADD DOCSTRING!!
    """

    def _solve_cycle(self, instance):
        if instance.num_carriers > 1:
            self.run_auction(instance)
        carrier = True
        while carrier:
            carrier = self.build_routes(instance)
            if carrier:
                self.initialize_another_tour(carrier)

    def run_auction(self, instance):
        au.Auction_b().execute(instance)

    def initialize_another_tour(self, carrier):
        EarliestDueDate().initialize_carrier(carrier)

    def build_routes(self, instance):
        carrier = I1Insertion().solve_instance(instance)
        return carrier


class DynamicI1InsertionWithAuctionC(DynamicSolver):
    """
    ADD DOCSTRING!!
    """

    def _solve_cycle(self, instance):
        if instance.num_carriers > 1:
            self.run_auction(instance)
        carrier = True
        while carrier:
            carrier = self.build_routes(instance)
            if carrier:
                self.initialize_another_tour(carrier)

    def run_auction(self, instance):
        au.Auction_c().execute(instance)

    def initialize_another_tour(self, carrier):
        EarliestDueDate().initialize_carrier(carrier)

    def build_routes(self, instance):
        carrier = I1Insertion().solve_instance(instance)
        return carrier
