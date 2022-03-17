import logging
import random
from abc import ABC, abstractmethod

from core_module import instance as it, solution as slt

logger = logging.getLogger(__name__)


def _abs_num_requests(carrier_: slt.AHDSolution, num_submitted_requests) -> int:
    """
    returns the absolute number of requests that a carrier shall submit, depending on whether it was initially
    given as an absolute int or a float (relative)
    """
    if isinstance(num_submitted_requests, int):
        return num_submitted_requests
    elif isinstance(num_submitted_requests, float):
        if num_submitted_requests % 1 == 0:
            return int(num_submitted_requests)
        assert num_submitted_requests <= 1, 'If providing a float, must be <=1 to be converted to percentage'
        return round(len(carrier_.assigned_requests) * num_submitted_requests)


class RequestSelectionBehavior(ABC):
    def __init__(self, num_submitted_requests: int):
        self.num_submitted_requests = num_submitted_requests
        self.name = self.__class__.__name__

    @abstractmethod
    def execute(self, instance: it.CAHDInstance, solution: slt.CAHDSolution):
        pass


# =====================================================================================================================
# REQUEST SELECTION BASED ON INDIVIDUAL REQUEST EVALUATION
# =====================================================================================================================


# =====================================================================================================================
# NEIGHBOR-BASED REQUEST SELECTION
# =====================================================================================================================


# =====================================================================================================================
# BUNDLE-BASED REQUEST SELECTION
# =====================================================================================================================


# class TimeShiftCluster(RequestSelectionBehaviorCluster):
#     """Selects the cluster that yields the highest temporal flexibility when removed"""
#     pass

class InfeasibleFirstRandomSecond(RequestSelectionBehavior):
    """
    Those requests that got accepted although they were infeasible will be selected.
    If more requests than that can be submitted, the remaining ones are selected randomly
    """

    def execute(self, instance: it.CAHDInstance, solution: slt.CAHDSolution):
        auction_request_pool = []
        original_partition_labels = []
        for carrier in solution.carriers:
            k = _abs_num_requests(carrier, self.num_submitted_requests)
            selected = random.sample(carrier.accepted_infeasible_requests,
                                     min(k, len(carrier.accepted_infeasible_requests)))
            k -= len(selected)
            if k > 0:
                valuations = []
                for request in carrier.accepted_requests:
                    valuation = self._evaluate_request(instance, solution, carrier, request)
                    valuations.append(valuation)

                # pick the WORST k evaluated requests (from ascending order)
                selected += [r for _, r in sorted(zip(valuations, carrier.accepted_requests))][:k]
                selected.sort()

            for request in selected:
                solution.free_requests_from_carriers(instance, [request])
                auction_request_pool.append(request)
                original_partition_labels.append(carrier.id_)

        return auction_request_pool, original_partition_labels

    def _evaluate_request(self, instance: it.CAHDInstance, solution: slt.CAHDSolution, carrier: slt.AHDSolution,
                          request: int):
        return random.random()
