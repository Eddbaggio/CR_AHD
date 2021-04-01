import abc
import random


class TWSelectionBehavior(abc.ABC):
    def execute(self, tw_offer_set):
        selected_tw = self.select_tw(tw_offer_set)
        return selected_tw

    @abc.abstractmethod
    def select_tw(self, tw_offer_set):
        pass


class UniformPreference(TWSelectionBehavior):
    """Will randomly select a TW """

    def select_tw(self, tw_offer_set):
        return random.choice(tw_offer_set)


class EarlyPreference(TWSelectionBehavior):
    """Will always select the earliest TW available based on the time window opening"""

    def select_tw(self, tw_offer_set):
        return min(tw_offer_set, key=lambda tw: tw.e)


class LatePreference(TWSelectionBehavior):
    """Will always select the latest TW available based on the time window closing"""

    def select_tw(self, tw_offer_set):
        return max(tw_offer_set, key=lambda tw: tw.l)

# class PreferEarlyAndLate(TWSelectionBehavior):
#     def select_tw(self, tw_offer_set):
#         pass
