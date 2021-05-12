import abc
import logging
from copy import deepcopy

from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import tour_construction as cns
from src.cr_ahd.tw_management_module import tw_offering as two, tw_selection as tws

logger = logging.getLogger(__name__)


class TWManagement(abc.ABC):
    def execute(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        for carrier in range(instance.num_carriers):

            # need a temp copy of the carrier to get TW offerings for *multiple* requests (which i need due to the way
            # the cycles are structured: assign multiple requests, offer tws to all of them, do the auction, ...)
            tmp_carrier_ = deepcopy(solution.carriers[carrier])
            solution.carriers.append(tmp_carrier_)
            tmp_carrier = instance.num_carriers

            for request in solution.carriers[tmp_carrier].unrouted_requests:

                offer_set = self._get_offer_set(instance, solution, tmp_carrier, request)
                logger.debug(f'time windows{offer_set} are offered to request {request} by carrier {carrier}')

                selected_tw = self._get_selected_tw(offer_set, request)
                logger.debug(f'time window {selected_tw} was chosen by request {request}')

                pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(request)
                solution.tw_open[delivery_vertex] = selected_tw.open
                solution.tw_close[delivery_vertex] = selected_tw.close
                # execute the insertion. this must be done in twm since twm is done in batches
                pdp_insertion = cns.CheapestPDPInsertion()
                insertion_operation = pdp_insertion._carrier_cheapest_insertion(instance, solution, tmp_carrier, [request])

                if insertion_operation[1] is None:
                    pdp_insertion._create_new_tour_with_request(instance, solution, tmp_carrier, request)

                else:
                    pdp_insertion._execute_insertion(instance, solution, tmp_carrier, *insertion_operation)

            # pop the temp carrier from the solution:
            solution.carriers.pop()
        pass

    @abc.abstractmethod
    def _get_offer_set(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, request: int):
        pass

    @abc.abstractmethod
    def _get_selected_tw(self, offer_set, request: int):
        pass


class TWManagement0(TWManagement):
    """carrier: offer all feasible time windows, customer: select a random time window from the offered set"""
    def _get_offer_set(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, request: int):
        return two.FeasibleTW().execute(instance, solution, carrier, request)

    def _get_selected_tw(self, offer_set, request: int):
        return tws.UniformPreference().execute(offer_set, request)
