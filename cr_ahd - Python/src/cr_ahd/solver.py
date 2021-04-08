import abc
import logging.config
from typing import final

import src.cr_ahd.tw_management_module.tw_management as twm
import src.cr_ahd.utility_module.utils as ut
from src.cr_ahd.auction_module import auction as au, \
    bidding as bd, \
    bundle_generation as bg, \
    request_selection as rs, \
    winner_determination as wd
from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import tour_construction as cns, tour_improvement as imp, tour_initialization as ini

logger = logging.getLogger(__name__)


class Solver(abc.ABC):
    def execute(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        """
        apply the concrete solution algorithm
        :param solution:
        :param instance:
        :return: None
        """
        self._solve(instance, solution)
        solution.solution_algorithm = self
        pass

    @abc.abstractmethod
    def _solve(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        pass


class StaticSequentialInsertion(Solver):
    """
    Initializes pendulum tours first! Then builds the routes. The Dynamic Version does NOT initialize tours
    """

    def _solve(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
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
        ini.EarliestDueDate().initialize_instance(instance)  # not flexible!
        pass

    def build_routes(self, instance):
        cns.SequentialInsertion().solve(instance)
        pass

    def finalize_with_local_search(self, instance):
        imp.TwoOpt().improve_instance(instance)
        pass


class StaticI1Insertion(Solver):
    def _solve(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
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
        ini.EarliestDueDate().initialize_instance(instance)  # not flexible!
        pass

    def initialize_another_tour(self, carrier):
        ini.EarliestDueDate().initialize_carrier(carrier)

    def build_routes(self, instance):
        carrier = cns.I1Insertion().solve(instance)
        return carrier

    def finalize_with_local_search(self, instance):
        imp.TwoOpt().improve_instance(instance)
        pass


class StaticI1InsertionWithAuction(Solver):
    def _solve(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
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
        ini.EarliestDueDate().initialize_carrier(carrier)

    def build_routes(self, instance):
        carrier = cns.I1Insertion().solve(instance)
        return carrier

    def finalize_with_local_search(self, instance):
        imp.TwoOpt().improve_instance(instance)


# =====================================================================================================================
# DYNAMIC
# =====================================================================================================================


class DynamicSolver(Solver):
    @final
    def _solve(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        while solution.unassigned_requests:
            self.assign_n_requests(instance, solution, ut.DYNAMIC_CYCLE_TIME)
            self._solve_cycle(instance, solution)
        self.finalize_with_local_search(instance, solution)

    @final
    def assign_n_requests(self, instance: it.PDPInstance, solution: slt.GlobalSolution, n):
        """Assigns n REQUESTS to their corresponding carrier, following the data stored in the instance.
        Assumes that each carrier has the same num_request!
        Does not work on a vertex level but on a request level!
        """
        logger.debug(f'Assigning {n} requests to each carrier')
        # TODO assumes equal num_requests for all carriers! Does not work otherwise!
        assert len(solution.unassigned_requests) % instance.num_carriers == 0
        k: int = len(solution.unassigned_requests) // instance.num_carriers
        requests = []
        for c in range(instance.num_carriers):
            for i in range(n):
                requests.append(solution.unassigned_requests[c * k + i])
        carriers = [instance.request_to_carrier_assignment[i] for i in requests]
        solution.assign_requests_to_carriers(requests, carriers)
        pass

    @final
    def finalize_with_local_search(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        # imp.TwoOpt().improve_instance(instance)
        pass

    @abc.abstractmethod
    def _solve_cycle(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        pass


class DynamicSequentialInsertion(DynamicSolver):
    """
    Does NOT initialize pendulum tours!
    """

    def _solve_cycle(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        cns.SequentialInsertion().solve(instance, solution)
        pass


class DynamicCheapestInsertion(DynamicSolver):

    def _solve_cycle(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        cns.CheapestInsertion().solve(instance, solution)
        pass


class DynamicI1Insertion(DynamicSolver):
    """
    ADD DOCSTRING!!
    """

    def _solve_cycle(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        self.build_routes()
        '''
        carrier = True
        while carrier:
            carrier = self.build_routes(instance)
            if carrier:
                self.initialize_another_tour(carrier)
        '''

    def initialize_another_tour(self, carrier):
        ini.EarliestDueDate().initialize_carrier(carrier)

    def build_routes(self, instance, solution):
        cns.I1Insertion().solve(instance, solution)
        pass


class DynamicI1InsertionWithAuctionA(DynamicSolver):
    """
    ADD DOCSTRING!!
    """

    def _solve_cycle(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
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
        ini.EarliestDueDate().initialize_carrier(carrier)

    def build_routes(self, instance):
        carrier = cns.I1Insertion().solve(instance)
        return carrier


class DynamicI1InsertionWithAuctionB(DynamicSolver):
    """
    ADD DOCSTRING!!
    """

    def _solve_cycle(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
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
        ini.EarliestDueDate().initialize_carrier(carrier)

    def build_routes(self, instance):
        carrier = cns.I1Insertion().solve(instance)
        return carrier


class DynamicI1InsertionWithAuctionC(DynamicSolver):
    """
    ADD DOCSTRING!!
    """

    def _solve_cycle(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
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
        ini.EarliestDueDate().initialize_carrier(carrier)

    def build_routes(self, instance):
        carrier = cns.I1Insertion().solve(instance)
        return carrier


class DynamicCollaborativeAHD(Solver):
    """
    The full program: Dynamic TW Management and auction-based Collaboration. Uses Auction type C
    """

    # TODO refactoring and tidy up required! Can I create a meaningful superclass?

    @final
    def _solve(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        while instance.unrouted_requests:
            self.assign_n_requests(instance, ut.DYNAMIC_CYCLE_TIME)
            self._solve_cycle(instance)
        self.finalize_with_local_search(instance)

    @final
    def assign_n_requests(self, instance, n):
        """from the unrouted requests present in the instance, assign n requests to their corresponding carrier as
        defined by the Vertex.carrier_assignment attribute"""
        logger.debug(f'Assigning {n} requests to carriers')
        for request in instance.unassigned_requests()[:n]:
            request._tw = ut.TIME_HORIZON  # reset the time window that is defined by the solomon instance
            carrier = ut.get_carrier_by_id(instance.carriers, request.carrier_assignment)
            carrier.assign_requests([request])
            twm.TWManagement1().execute(carrier, request)

    @final
    def finalize_with_local_search(self, instance):
        imp.TwoOpt().improve_instance(instance)
        pass

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
        ini.EarliestDueDate().initialize_carrier(carrier)

    def build_routes(self, instance):
        carrier = cns.I1Insertion().solve(instance)
        return carrier
