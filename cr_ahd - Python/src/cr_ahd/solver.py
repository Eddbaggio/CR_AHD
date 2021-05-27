import abc
import logging.config
from typing import final

from src.cr_ahd.utility_module import utils as ut, plotting as pl
from src.cr_ahd.auction_module import auction as au
from src.cr_ahd.tw_management_module import tw_management as twm
from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import tour_construction as cns, tour_improvement as imp, tour_initialization as ini

logger = logging.getLogger(__name__)


class Solver(abc.ABC):
    def execute(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        """
        apply the concrete solution algorithm
        """
        self._solve(instance, solution)
        solution.solution_algorithm = self.__class__.__name__
        pass

    @abc.abstractmethod
    def _solve(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        pass


class StaticSolver(Solver):
    @final
    def _solve(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        # assign all requests to their corresponding carrier
        self.assign_all_requests(instance, solution)

        # skip auction for centralized instances
        if instance.num_carriers > 1:
            self.run_auction(instance, solution)

        # build tours with the re-allocated requests
        self.initialize_pendulum_tours(instance, solution)
        self.build_tours(instance, solution)
        self.finalize_with_local_search(instance, solution)

    pass

    def assign_all_requests(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        logger.debug(f'Assigning all requests to their carriers')
        solution.assign_requests_to_carriers(instance.requests, instance.request_to_carrier_assignment)
        self.time_window_management(instance, solution)
        pass

    @abc.abstractmethod
    def initialize_pendulum_tours(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        pass

    @abc.abstractmethod
    def build_tours(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        pass

    def run_auction(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        pass

    def finalize_with_local_search(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        imp.PDPMoveBestImpr().improve_global_solution(instance, solution)
        # imp.PDPTwoOptBest().improve_global_solution(instance, solution)
        # imp.PDPExchangeMoveBest().improve_global_solution(instance, solution)
        pass

    def time_window_management(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        pass


class Static(StaticSolver):
    def initialize_pendulum_tours(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        ini.FurthestDistance().execute(instance, solution)
        pass

    def build_tours(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        cns.CheapestPDPInsertion().construct(instance, solution)
        pass


class StaticCollaborative(StaticSolver):
    """
    Executes an auction after all the requests have been assigned
    """

    def initialize_pendulum_tours(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        ini.FurthestDistance().execute(instance, solution)
        pass

    def build_tours(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        cns.CheapestPDPInsertion().construct(instance, solution)

    def run_auction(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        au.AuctionD().execute(instance, solution)


class StaticCollaborativeAHD(StaticSolver):
    """
    Assigns all requests to the original carrier, then does the time window management and finally does an auction
    """

    def initialize_pendulum_tours(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        ini.FurthestDistance().execute(instance, solution)

    def build_tours(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        cns.CheapestPDPInsertion().construct(instance, solution)

    def run_auction(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        au.AuctionD().execute(instance, solution)

    def time_window_management(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        twm.TWManagementMultiple0().execute(instance, solution)


# =====================================================================================================================
# DYNAMIC
# =====================================================================================================================


def assign_n_requests(instance: it.PDPInstance, solution: slt.GlobalSolution, n):
    """Assigns n REQUESTS to their corresponding carrier, following the data stored in the instance.
    Assumes that each carrier has the same num_request!
    Does not work on a vertex level but on a request level!
    """
    logger.debug(f'Assigning {n} requests to each carrier')
    # assumes equal num_requests for all carriers! Does not work otherwise!
    assert len(solution.unassigned_requests) % instance.num_carriers == 0
    assert n <= instance.num_requests / instance.num_carriers
    k: int = len(solution.unassigned_requests) // instance.num_carriers
    requests = []
    for c in range(instance.num_carriers):
        for i in range(min(n, k)):
            requests.append(solution.unassigned_requests[c * k + i])
    carriers = [instance.request_to_carrier_assignment[i] for i in requests]
    solution.assign_requests_to_carriers(requests, carriers)
    pass


class DynamicSolver(Solver, abc.ABC):
    @final
    def _solve(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        while solution.unassigned_requests:
            assign_n_requests(instance, solution, ut.DYNAMIC_CYCLE_TIME)
            self.time_window_management(instance, solution)
            if instance.num_carriers > 1:  # not for centralized instances
                self.run_auction(instance, solution)
            # build tours with the re-allocated requests
            self.build_tours(instance, solution)
        self.finalize_with_local_search(instance, solution)

    def time_window_management(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        pass

    def finalize_with_local_search(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        imp.PDPMoveBestImpr().improve_global_solution(instance, solution)
        # imp.PDPExchangeMoveBest().improve_global_solution(instance, solution)
        pass

    def run_auction(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        pass

    def build_tours(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        cns.CheapestPDPInsertion().construct(instance, solution)
        pass


class Dynamic(DynamicSolver):
    """
    requests arrive dynamically. upon request disclosure, it must be inserted into a tour immediately.
    No auction, to time windows
    """
    pass


class DynamicCollaborative(DynamicSolver):
    """
    requests arrive dynamically in specified cyclic intervals. Once some requests have been collected, an auction
    takes place and afterwards, allocated requests are inserted into a tour
    """

    def run_auction(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        au.AuctionD().execute(instance, solution)


class DynamicAHD(DynamicSolver):
    """
    requests arrive dynamically. Upon request disclosure, a time window management procedure is performed to determine
    the time window for each new request.
    """

    def time_window_management(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        twm.TWManagementMultiple0().execute(instance, solution)


class DynamicCollaborativeAHD(DynamicSolver):
    """
    The full program: Dynamic TW Management and auction-based Collaboration.
    """

    def time_window_management(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        twm.TWManagementMultiple0().execute(instance, solution)

    def run_auction(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        au.AuctionD().execute(instance, solution)


# =====================================================================================================================
# DYNAMIC 2
# =====================================================================================================================

class DynamicCollaborativeAHDSingleAuction(Solver):
    """
    TWM is done one request at a time, i.e. the way it's supposed to be done.
    Only a single auction after the acceptance phase
    """

    def _solve(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        while solution.unassigned_requests:
            # assign the next request
            request = solution.unassigned_requests[0]
            carrier = instance.request_to_carrier_assignment[request]
            solution.assign_requests_to_carriers([request], [carrier])

            # find the tw for the request
            twm.TWManagementSingle().execute(instance, solution, carrier)

            # build tours with the assigned and tw-managed requests
            cns.CheapestPDPInsertion().construct(instance, solution)
            imp.PDPMoveBestImpr().improve_global_solution(instance, solution)

        # unroute all requests and run a single auction at the end
        for carrier_ in solution.carriers:
            unrouted = solution.request_to_carrier_assignment[solution.request_to_carrier_assignment == carrier_.id_]
            carrier_.unrouted_requests = list(unrouted.astype(int))
            carrier_.tours.clear()

        if instance.num_carriers > 1:  # not for centralized instances
            au.AuctionD().execute(instance, solution)
            # TODO: apparently the requests do not get reassigned properly!

        # build tours after the auction
        cns.CheapestPDPInsertion().construct(instance, solution)
        imp.PDPMoveBestImpr().improve_global_solution(instance, solution)
