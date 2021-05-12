import abc
import logging
import random
from typing import List

from src.cr_ahd.utility_module.utils import TimeWindow

logger = logging.getLogger(__name__)


class TWSelectionBehavior(abc.ABC):
    # TODO maybe in the future, i have to store also time window preferences / tw selection behavior in the instance
    def execute(self, tw_offer_set: List[TimeWindow], request: int):
        selected_tw = self.select_tw(tw_offer_set, request)
        return selected_tw

    @abc.abstractmethod
    def select_tw(self, tw_offer_set, request: int):
        pass


class UniformPreference(TWSelectionBehavior):
    """Will randomly select a TW """
    def select_tw(self, tw_offer_set, request: int):
        return random.choice(tw_offer_set)


class EarlyPreference(TWSelectionBehavior):
    """Will always select the earliest TW available based on the time window opening"""
    def select_tw(self, tw_offer_set, request: int):
        return min(tw_offer_set, key=lambda tw: tw.open)


class LatePreference(TWSelectionBehavior):
    """Will always select the latest TW available based on the time window closing"""
    def select_tw(self, tw_offer_set, request: int):
        return max(tw_offer_set, key=lambda tw: tw.close)

# class PreferEarlyAndLate(TWSelectionBehavior):
#     def select_tw(self, tw_offer_set):
#         pass
