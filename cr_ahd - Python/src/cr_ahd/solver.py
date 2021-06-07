import abc
import logging.config
import random
from copy import deepcopy
from typing import final

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

        self._acceptance_phase(instance, solution)
        self._auction_phase(instance, solution)

        solution.solution_algorithm = self.__class__.__name__
        return solution

    def _acceptance_phase(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        while solution.unassigned_requests:
            # assign the next request
            request = solution.unassigned_requests[0]
            carrier = instance.request_to_carrier_assignment[request]
            solution.assign_requests_to_carriers([request], [carrier])

            # find the tw for the request
            self._time_window_management(instance, solution, carrier)

            # build tours with the assigned request
            cns.CheapestPDPInsertion().construct(instance, solution)
            imp.PDPMoveBestImpr().improve_global_solution(instance, solution)

    @abc.abstractmethod
    def _time_window_management(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        pass

    def _auction_phase(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        """
        includes request selection, bundle generation, bidding, winner determination and also the final routing
        after the auction
        :param instance:
        :param solution:
        :return:
        """
        pass


class IsolatedPlanning(Solver):
    def _time_window_management(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        twm.TWManagementSingle().execute(instance, solution, carrier)


class CollaborativePlanning(Solver):
    """
    TWM is done one request at a time, i.e. the way it's supposed to be done.
    Only a single auction after the acceptance phase
    """

    def _auction_phase(self, instance: it.PDPInstance, solution: slt.CAHDSolution):

        if instance.num_carriers > 1:  # not for centralized instances
            au.AuctionD().execute(instance, solution)

        # unroute all requests to imitate acceptance phase dynamism
        for carrier_ in solution.carriers:
            carrier_.tours.clear()
            carrier_.unrouted_requests = carrier_.assigned_requests

        # do the final, dynamic (!) route construction. Must be dynamic (as in acceptance phase) to be guaranteed to
        # find at least the same solutions as in acceptance phase
        construction = cns.CheapestPDPInsertion()
        for carrier in range(instance.num_carriers):
            carrier_ = solution.carriers[carrier]
            while carrier_.unrouted_requests:
                insertion = construction._carrier_cheapest_insertion(instance, solution, carrier,
                                                                     carrier_.unrouted_requests[:1])

                request, tour, pickup_pos, delivery_pos = insertion

                # when for a given request no tour can be found, create a new tour and start over. This may raise
                # a ConstraintViolationError if the carrier cannot initialize another new tour
                if tour is None:
                    construction._create_new_tour_with_request(instance, solution, carrier, request)

                else:
                    construction._execute_insertion(instance, solution, carrier, request, tour, pickup_pos,
                                                    delivery_pos)

                imp.PDPMoveBestImpr().improve_carrier_solution_first_improvement(instance, solution, carrier)

    def _time_window_management(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        twm.TWManagementSingle().execute(instance, solution, carrier)


class CentralizedPlanning(Solver):

    def execute(self, instance: it.PDPInstance):

        # copy and alter the underlying instance to make it a multi-depot, single-carrier instance
        md_instance = deepcopy(instance)
        md_instance.num_carriers = 1
        md_instance.request_to_carrier_assignment = [0] * len(md_instance.request_to_carrier_assignment)

        solution = slt.CAHDSolution(md_instance)
        random.seed(0)

        self._acceptance_phase(md_instance, solution)

        solution.solution_algorithm = self.__class__.__name__
        return solution

    def _acceptance_phase(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        while solution.unassigned_requests:
            request = solution.unassigned_requests[0]
            carrier = instance.request_to_carrier_assignment[request]
            solution.assign_requests_to_carriers([request], [carrier])

            # find the tw for the request
            self._time_window_management(instance, solution, 0)

            # build tours with the assigned request
            cns.CheapestPDPInsertion().construct(instance, solution)
            imp.PDPMoveBestImpr().improve_global_solution(instance, solution)
        pass

    def _time_window_management(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        twm.TWManagementSingle().execute(instance, solution, carrier)


class IsolatedPlanningNoTW(IsolatedPlanning):
    def _time_window_management(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        pass


class CollaborativePlanningNoTW(CollaborativePlanning):
    def _time_window_management(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        pass
