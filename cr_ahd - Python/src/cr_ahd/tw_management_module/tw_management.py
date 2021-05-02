import abc
from src.cr_ahd.core_module import instance as it, solution as slt
import src.cr_ahd.tw_management_module.tw_offering as two
import src.cr_ahd.tw_management_module.tw_selection as tws
import src.cr_ahd.utility_module.utils as ut


class TWManagement(abc.ABC):
    def execute(self, instance: it.PDPInstance, solution: slt.GlobalSolution):
        for carrier in range(instance.num_carriers):
            for request in solution.carriers[carrier].unrouted_requests:
                offer_set = self.get_offer_set(instance, solution, carrier, request)
                selected_tw = self.get_selected_tw(offer_set, request)
                pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(request)
                solution.tw_open[delivery_vertex] = selected_tw.open
                solution.tw_close[delivery_vertex] = selected_tw.close
        pass

    @abc.abstractmethod
    def get_offer_set(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, request: int):
        pass

    @abc.abstractmethod
    def get_selected_tw(self, offer_set, request: int):
        pass


class TWManagement0(TWManagement):
    """carrier: offer all feasible time windows, customer: select a random time window"""
    def get_offer_set(self, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, request: int):
        two.FeasibleTW().execute(instance, solution, carrier, request)

    def get_selected_tw(self, offer_set, request: int):
        tws.UniformPreference().execute(offer_set, request)
