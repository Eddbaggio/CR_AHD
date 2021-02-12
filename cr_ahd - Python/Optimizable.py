from abc import abstractmethod


class Optimizable(object):
    """Superclass for all subclasses that can be optimized, i.e. instance, carrier, tour"""
    @property
    @abstractmethod
    def distance_matrix(self):
        return

    # @abstractmethod
    # not applicable for a tour
    def initialize(self, visitor):
        pass
