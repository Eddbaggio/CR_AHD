import abc
import src.cr_ahd.tw_management_module.tw_offering as two
import src.cr_ahd.tw_management_module.tw_selection as tws
import src.cr_ahd.utility_module.utils as ut


class TWManagement(abc.ABC):
    def execute(self, carrier, request):
        # assert request.tw == ut.TIME_HORIZON
        offer_set = self.get_offer_set(carrier, request)
        selected_tw = self.get_selected_tw(offer_set)
        request.tw = selected_tw
        pass

    @abc.abstractmethod
    def get_offer_set(self, carrier, request):
        pass

    @abc.abstractmethod
    def get_selected_tw(self, offer_set):
        pass


class TWManagement1(TWManagement):
    def get_offer_set(self, carrier, request):
        return two.FeasibleTW().execute(carrier, request)

    def get_selected_tw(self, offer_set):
        return tws.EarlyPreference().execute(offer_set)
