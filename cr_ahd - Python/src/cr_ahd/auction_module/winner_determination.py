from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np

from src.cr_ahd.core_module.carrier import Carrier
from src.cr_ahd.core_module.vertex import Vertex


class WinnerDeterminationBehavior(ABC):
    def execute(self, bids: Dict[Tuple[Vertex], Dict[Carrier, float]]):
        """
        apply the concrete winner determination behavior. NOTE: this will modify the bids and render them useless!

        :param bids: dict of bids per bundle per carrier: {bundleA: {carrier1: bid, carrier2: bid}, bundleB: {
        carrier1: bid, carrier2: bid}
        :return: nothing for now
        """

        for bundle, carrier_bids in bids.items():
            winner, winning_bid = self._determine_winner(carrier_bids)
            winner.assign_requests(bundle)
            self._remove_bids_of_carrier(winner, bids)  # updating the dict I'm iterating over
        return

    @abstractmethod
    def _determine_winner(self, carrier_bids):
        pass

    def _remove_bids_of_carrier(self, carrier, all_bids):
        """

        :param carrier: the carrier whose bids shall be removed from the given bids
        :param all_bids: set of ALL bids (dict of dicts) from which to remove the ones submitted by carrier
        :return:
        """
        for bundle, bundle_bids in all_bids.items():
            bundle_bids[carrier] = np.infty
        pass


class LowestBid(WinnerDeterminationBehavior):
    def _determine_winner(self, carrier_bids):
        """

        :param carrier_bids: dict of {carrierA: bidA, carrierB: bidB, ... }
        :return:
        """
        return min(carrier_bids.items(), key=lambda x: x[1])
