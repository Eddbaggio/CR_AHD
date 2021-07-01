import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Sequence

import tqdm

from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import tour_construction as cns, local_search as imp, tour_initialization as ini
from src.cr_ahd.utility_module.utils import ConstraintViolationError

logger = logging.getLogger(__name__)


class BiddingBehavior(ABC):
    def execute(self,
                instance: it.PDPInstance,
                solution: slt.CAHDSolution,
                bundles: Sequence[Sequence[int]]) -> List[List[float]]:
        """
        returns a nested list of bids. the first axis is the bundles, the second axis (inner lists) contain the carrier
        bids on that bundle:

        """

        bundle_bids = []
        for b in tqdm.trange(len(bundles), desc='Bidding', disable=True):
            carrier_bundle_bids = []
            for carrier in range(instance.num_carriers):
                logger.debug(f'Carrier {carrier} generating bids for bundle {b}')

                value_without_bundle = self._value_without_bundle(instance, solution, carrier)

                # create & append a temporary copy of the carrier which will be used to compute the bid
                tmp_carrier, tmp_carrier_ = self._create_tmp_carrier_copy(instance, solution, bundles[b], carrier)

                value_with_bundle = self._value_with_bundle(instance, solution, bundles[b], tmp_carrier)
                bid = value_with_bundle - value_without_bundle
                carrier_bundle_bids.append(bid)

                solution.carriers.pop()  # del the temporary carrier copy
                solution.carrier_depots.pop()  # del the tmp_carrier's depot

            bundle_bids.append(carrier_bundle_bids)
        solution.bidding_behavior = self.__class__.__name__
        return bundle_bids

    def _create_tmp_carrier_copy(self, instance: it.PDPInstance, solution: slt.CAHDSolution, bundle: Sequence[int],
                                 carrier: int):
        # create a temporary copy to compute the correct bids
        tmp_carrier = instance.num_carriers
        tmp_carrier_ = deepcopy(solution.carriers[carrier])

        # add the bundle to the carriers assigned and accepted requests
        tmp_carrier_.assigned_requests.extend(bundle)
        tmp_carrier_.assigned_requests.sort()

        tmp_carrier_.accepted_requests.extend(bundle)
        tmp_carrier_.accepted_requests.sort()

        # reset the temporary carrier's solution and start from scratch instead
        tmp_carrier_.clear_routes()

        solution.carriers.append(tmp_carrier_)
        solution.carrier_depots.append(solution.carrier_depots[carrier])

        return tmp_carrier, tmp_carrier_

    @abstractmethod
    def _value_without_bundle(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        pass

    @abstractmethod
    def _value_with_bundle(self, instance: it.PDPInstance, solution: slt.CAHDSolution, bundle: Sequence[int],
                           tmp_carrier: int):
        pass


class StaticProfit(BiddingBehavior):
    def _value_without_bundle(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):

        return solution.carriers[carrier].sum_profit()

    def _value_with_bundle(self, instance: it.PDPInstance, solution: slt.CAHDSolution, bundle: Sequence[int],
                           tmp_carrier: int):
        tmp_carrier_ = solution.carriers[tmp_carrier]
        construction = cns.CheapestPDPInsertion()

        try:
            ini.FurthestDistance()._initialize_carrier(instance, solution, tmp_carrier)
            while tmp_carrier_.unrouted_requests:
                insertion = construction.best_insertion_for_carrier(instance,
                                                                    solution,
                                                                    tmp_carrier,
                                                                    solution.carriers[tmp_carrier].unrouted_requests)
                request, tour, pickup_pos, delivery_pos = insertion

                # when for a given request no tour can be found, create a new tour. This may raise a
                # ConstraintViolationError if the carrier (a) would exceed fleet size constraints or (b) a pendulum
                # tour would exceed time window constraints

                if tour is None:
                    construction.create_new_tour_with_request(instance, solution, tmp_carrier, request)

                else:
                    construction.execute_insertion(instance, solution, tmp_carrier, request, tour, pickup_pos,
                                                   delivery_pos)

            # local search
            imp.PDPMove().improve_carrier_solution(instance, solution, tmp_carrier, False)
            imp.PDPRelocate().improve_carrier_solution(instance, solution, tmp_carrier, False)
            imp.PDPTwoOpt().improve_carrier_solution(instance, solution, tmp_carrier, False)

            with_bundle = solution.carriers[tmp_carrier].sum_profit()

        except ConstraintViolationError as e:
            with_bundle = -float('inf')

        return with_bundle
