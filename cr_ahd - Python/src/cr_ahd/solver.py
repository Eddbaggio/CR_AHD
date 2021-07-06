import abc
import logging.config
import random
from copy import deepcopy

import src.cr_ahd.utility_module.utils as ut
from src.cr_ahd.auction_module import auction as au
from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import tour_construction as cns, tour_initialization as ini, metaheuristics as mh
from src.cr_ahd.tw_management_module import tw_management as twm

logger = logging.getLogger(__name__)


class Solver(abc.ABC):
    # TODO include starting_solution
    def execute(self, instance: it.PDPInstance, starting_solution: slt.CAHDSolution = None):
        """
        apply the concrete solution algorithm
        """
        if starting_solution is None:
            solution = slt.CAHDSolution(instance)
        else:
            solution = starting_solution

        random.seed(0)

        solution = self._acceptance_phase(instance, solution)
        # static routing is not required if request selection is route-independent (such as in the cluster RS)?
        # solution = self._static_routing(instance, solution)
        mh.PDPVariableNeighborhoodDescent().execute(instance, solution)
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
            accepted = self._time_window_management(instance, solution, carrier, request)

            # build tours with the assigned request if it was accepted
            if accepted:
                # solution = cns.MinTimeShiftInsertion().construct_dynamic(instance, solution, carrier)
                cns.MinTravelDistanceInsertion().construct_dynamic(instance, solution, carrier)

        ut.validate_solution(instance, solution)
        return solution

    def _time_window_management(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, request: int):
        return twm.TWManagementSingle().execute(instance, solution, carrier, request)

    def _static_routing(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        raise RuntimeError('static routing is omitted atm since it does not yield improvements over the dynamic routes')
        solution = deepcopy(solution)
        solution.clear_carrier_routes()

        # create seed tours
        ini.MaxCliqueTourInitialization().execute(instance, solution)

        # construct_static initial solution
        # solution = cns.MinTimeShiftInsertion().construct_static(instance, solution)
        cns.MinTravelDistanceInsertion().construct_static(instance, solution)

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
    pass


class CollaborativePlanning(Solver):
    """
    TWM is done one request at a time, i.e. the way it's supposed to be done.
    Only a single auction after the acceptance phase
    """

    def _auction_phase(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        solution = deepcopy(solution)
        if instance.num_carriers > 1:  # not for centralized instances
            au.AuctionD().execute(instance, solution)

        # do a final dynamic re-optimization from scratch
        solution.clear_carrier_routes()
        for carrier in range(solution.num_carriers()):
            while solution.carriers[carrier].unrouted_requests:
                cns.MinTravelDistanceInsertion().construct_dynamic(instance, solution, carrier)
            mh.PDPVariableNeighborhoodDescent().execute(instance, solution, [carrier])
        return solution


class CentralizedPlanning(Solver):

    def execute(self, instance: it.PDPInstance, starting_solution: slt.CAHDSolution = None):
        # copy and alter the underlying instance to make it a multi-depot, single-carrier instance
        md_instance = deepcopy(instance)
        md_instance.num_carriers = 1
        md_instance.carrier_depots = [[d for d in range(instance.num_depots)]]
        md_instance.request_to_carrier_assignment = [0] * len(md_instance.request_to_carrier_assignment)

        # initialize and adjust the solution
        solution = slt.CAHDSolution(md_instance)
        solution.carrier_depots = [[depot for depot in range(instance.num_depots)]]

        random.seed(0)  # TODO what is this for?

        solution = self._acceptance_phase(md_instance, solution)
        mh.PDPVariableNeighborhoodDescent().execute(instance, solution)
        # self._static_routing(md_instance, solution)

        solution.solution_algorithm = self.__class__.__name__
        return solution

    def _time_window_management(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, request: int):
        # use only the single, centralized carrier
        carrier = 0
        return twm.TWManagementSingle().execute(instance, solution, carrier, request)


class IsolatedPlanningNoTW(IsolatedPlanning):
    def _time_window_management(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, request: int):
        pass


class CollaborativePlanningNoTW(CollaborativePlanning):
    def _time_window_management(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int, request: int):
        pass
