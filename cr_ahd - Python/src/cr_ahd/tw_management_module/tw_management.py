import datetime as dt
import logging
from typing import Sequence, List

from src.cr_ahd.core_module import instance as it, solution as slt, tour as tr
from src.cr_ahd.tw_management_module import tw_offering as two, tw_selection as tws
from src.cr_ahd.utility_module import utils as ut

logger = logging.getLogger(__name__)


class TWManagementSingle:
    """
    handles a single request/customer at a time. a new routing is required after each call to this class'
    execute function! Requests must also be assigned one at a time
    """

    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        carrier_ = solution.carriers[carrier]
        assert len(carrier_.unrouted_requests) == 1, f'For the "Single" version of the TWM, only one request can be' \
                                                     f'handled at a time'
        request = carrier_.unrouted_requests[0]
        offer_set = two.FeasibleTW().execute(instance, solution, carrier, request)  # which TWs to offer?
        if offer_set:
            selected_tw = tws.UnequalPreference().execute(offer_set, request)  # which TW is selected?

            # set the TW open and close times
            pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(request)
            solution.tw_open[delivery_vertex] = selected_tw.open
            solution.tw_close[delivery_vertex] = selected_tw.close

        # in case no feasible TW exists for a given request
        else:
            logger.error(f'No feasible TW can be offered from Carrier {carrier} to request {request}')
            solution.rejected_requests.append(request)
            raise ut.ConstraintViolationError(
                f'No feasible TW can be offered from Carrier {carrier} to request {request}')
