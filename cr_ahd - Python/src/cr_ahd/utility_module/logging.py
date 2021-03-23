from abc import ABC, abstractmethod


class Observer(ABC):
    """
    The Observer interface declares the update method, used by subjects.
    """

    @abstractmethod
    def update(self, publisher) -> None:
        """
        Receive update from publisher.
        """
        pass

# class TourObserver(Observer):
#     def update(self, publisher) -> None:
#         if publisher

# tODO: would it be better to observe the Solver class instead? OR: observe both individually because they track
#  different things!!! individual responsibilities! Then maybe later merge them if that's possible -> tricky: i cannot
#  recreate the temporal order (unless i would timestamp each event)
