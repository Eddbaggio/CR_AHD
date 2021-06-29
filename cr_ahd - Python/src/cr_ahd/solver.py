import abc
import logging.config
import random
from copy import deepcopy
from typing import final
import src.cr_ahd.utility_module.utils as ut
from src.cr_ahd.auction_module import auction as au
from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import tour_construction as cns, tour_improvement as imp, tour_initialization as ini
from src.cr_ahd.tw_management_module import tw_management as twm

logger = logging.getLogger(__name__)


class Solver(abc.ABC):
    def execute(self, instance: it.PDPInstance):
        """
        apply the concrete solution algorithm
        """
        solution = slt.CAHDSolution(instance)
        random.seed(0)

        solution = self._acceptance_phase(instance, solution)
        # not required if request selection is route-independent (such as in the cluster RS)?
        solution = self._static_routing(instance, solution)
        solution = self._auction_phase(instance, solution)

        solution.solution_algorithm = self.__class__.__name__
        return solution

    def _acceptance_phase(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        solution = deepcopy(solution)
        while solution.unassigned_requests:
            # assign the next request
            request = solution.unassigned_requests[0]
            carrier = instance.request_to_carrier_assignment[request]
            solution.assign_requests_to_carriers([request], [carrier])

            # find the tw for the request
            self._time_window_management(instance, solution, carrier)

            # build tours with the assigned request
            cns.CheapestPDPInsertion().construct(instance, solution)
            imp.PDPMove().local_search(instance, solution)
            imp.PDPRelocate().local_search(instance, solution)
            imp.PDPTwoOpt().local_search(instance, solution)

        ut.validate_solution(instance, solution)
        return solution

    @abc.abstractmethod
    def _time_window_management(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        pass

    def _static_routing(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        solution = deepcopy(solution)
        solution.clear_carrier_routes()

        # create seed tours
        ini.MaxCliqueTourInitializationBehavior().execute(instance, solution)
        # construct initial solution
        cns.CheapestPDPInsertion().construct(instance, solution)
        # improve initial solution
        imp.PDPMove().local_search(instance, solution)
        imp.PDPRelocate().local_search(instance, solution)
        imp.PDPTwoOpt().local_search(instance, solution)

        ut.validate_solution(instance, solution)
        return solution

    def _auction_phase(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        """
        includes request selection, bundle generation, bidding, winner determination and also the final routing
        after the auction
        :param instance:
        :param solution:
        :return:
        """
        return solution


class IsolatedPlanning(Solver):
    def _time_window_management(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        twm.TWManagementSingle().execute(instance, solution, carrier)


class IsolatedPlanningNoReopt(Solver):
    def _time_window_management(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        twm.TWManagementSingle().execute(instance, solution, carrier)

    def _static_routing(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        return solution


class CollaborativePlanning(Solver):
    """
    TWM is done one request at a time, i.e. the way it's supposed to be done.
    Only a single auction after the acceptance phase
    """

    def _auction_phase(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        solution = deepcopy(solution)
        if instance.num_carriers > 1:  # not for centralized instances
            au.AuctionD().execute(instance, solution)

        # do a final, static routing
        self._static_routing(instance, solution)
        return solution

    def _time_window_management(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        twm.TWManagementSingle().execute(instance, solution, carrier)
        pass


class CentralizedPlanning(Solver):

    def execute(self, instance: it.PDPInstance):
        # copy and alter the underlying instance to make it a multi-depot, single-carrier instance
        md_instance = deepcopy(instance)
        md_instance.num_carriers = 1
        md_instance.carrier_depots = [[d for d in range(instance.num_depots)]]
        md_instance.request_to_carrier_assignment = [0] * len(md_instance.request_to_carrier_assignment)

        # initialize and adjust the solution
        solution = slt.CAHDSolution(md_instance)
        solution.carrier_depots = [[depot for depot in range(instance.num_depots)]]

        random.seed(0)

        self._acceptance_phase(md_instance, solution)
        self._static_routing(md_instance, solution)

        solution.solution_algorithm = self.__class__.__name__
        return solution

    def _time_window_management(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        # use only the single, centralized carrier
        carrier = 0
        twm.TWManagementSingle().execute(instance, solution, carrier)


class IsolatedPlanningNoTW(IsolatedPlanning):
    def _time_window_management(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        pass


class CollaborativePlanningNoTW(CollaborativePlanning):
    def _time_window_management(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        pass
