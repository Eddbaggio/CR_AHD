import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from src.cr_ahd.core_module import instance as it, solution as slt

logger = logging.getLogger(__name__)


class WinnerDeterminationBehavior(ABC):
    def execute(self, instance: it.PDPInstance, solution: slt.GlobalSolution, bundles: List, bids_matrix: List):
        """
        apply the concrete winner determination behavior. Each carrier can only win a single bundle for now

        """
        if not len(bundles) == instance.num_carriers:
            raise NotImplementedError('Proper set partitioning for WDP ist not implemented yet')
        bundle_winners = []
        for bundle, bids in zip(bundles, bids_matrix):
            winner, _ = self._determine_winner(bids)
            bundle_winners.append(winner)
            # ensure that the winner can not win another bundle by 'deleting' his bids (setting them to +- inf)
            self._remove_bids_of_carrier(winner, bids_matrix)
        return bundle_winners

    @abstractmethod
    def _determine_winner(self, carrier_bids) -> Tuple[int, float]:
        pass

    @abstractmethod
    def _remove_bids_of_carrier(self, carrier, all_bids):
        pass


class LowestBid(WinnerDeterminationBehavior):

    def _determine_winner(self, carrier_bids):
        """

        :param carrier_bids: dict of {carrierA: bidA, carrierB: bidB, ... }
        :return:
        """
        winner = np.argmin(carrier_bids)
        winning_bid = min(carrier_bids)
        return winner, winning_bid

    def _remove_bids_of_carrier(self, carrier: int, bids_matrix):
        for bundle_bids in bids_matrix:
            bundle_bids[carrier] = float('inf')
        pass
