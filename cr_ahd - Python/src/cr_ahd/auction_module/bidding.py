import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Sequence

import tqdm

from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.routing_module import tour_construction as cns, tour_initialization as ini, metaheuristics as mh
from src.cr_ahd.utility_module.utils import ConstraintViolationError

logger = logging.getLogger(__name__)


class BiddingBehavior(ABC):
    def __init__(self,
                 construction_method: cns.PDPParallelInsertionConstruction,
                 improvement_method: mh.PDPMetaHeuristic):
        self.construction_method = construction_method
        self.improvement_method = improvement_method

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

                value_without_bundle = solution.carriers[carrier].sum_profit()
                value_with_bundle = self._value_with_bundle(instance, solution, bundles[b], carrier)
                bid = value_with_bundle - value_without_bundle

                carrier_bundle_bids.append(bid)

            bundle_bids.append(carrier_bundle_bids)
        solution.bidding_behavior = self.__class__.__name__
        return bundle_bids

    def _create_tmp_carrier_copy_with_bundle(self, instance: it.PDPInstance, solution: slt.CAHDSolution,
                                             bundle: Sequence[int], carrier: int):
        # create a temporary copy to compute the correct bids
        tmp_carrier = instance.num_carriers  # ID
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
    def _value_with_bundle(self, instance: it.PDPInstance, solution: slt.CAHDSolution, bundle: Sequence[int],
                           carrier: int):
        pass


class DynamicReOpt(BiddingBehavior):
    def _value_with_bundle(self, instance: it.PDPInstance, solution: slt.CAHDSolution, bundle: Sequence[int],
                           carrier: int):

        # create temporary copy of the carrier
        tmp_carrier, tmp_carrier_ = self._create_tmp_carrier_copy_with_bundle(instance, solution, bundle, carrier)

        # assign and insert requests of the bundle
        try:
            while tmp_carrier_.unrouted_requests:
                self.construction_method.construct_dynamic(instance, solution, tmp_carrier)
            self.improvement_method.execute(instance, solution, [tmp_carrier])
            with_bundle = tmp_carrier_.sum_profit()

        except ConstraintViolationError:
            with_bundle = -float('inf')

        finally:
            solution.carriers.pop()  # del the temporary carrier copy
            solution.carrier_depots.pop()  # del the tmp_carrier's depot

        return with_bundle


class StaticProfit(BiddingBehavior):

    def _value_with_bundle(self, instance: it.PDPInstance, solution: slt.CAHDSolution, bundle: Sequence[int],
                           carrier: int):
        raise NotImplementedError('static routing does not find feasible solutions')
        # create & append a temporary copy of the carrier which will be used to compute the bid
        tmp_carrier, tmp_carrier_ = self._create_tmp_carrier_copy_with_bundle(instance, solution, bundle, carrier)

        try:
            ini.MaxCliqueTourInitialization()._initialize_carrier(instance, solution, tmp_carrier)
            self.construction_method.construct_static(instance, solution)
            self.improvement_method.execute(instance, solution, [tmp_carrier])
            with_bundle = solution.carriers[tmp_carrier].sum_profit()

        except ConstraintViolationError:
            with_bundle = -float('inf')

        finally:
            solution.carriers.pop()  # del the temporary carrier copy
            solution.carrier_depots.pop()  # del the tmp_carrier's depot

        return with_bundle
