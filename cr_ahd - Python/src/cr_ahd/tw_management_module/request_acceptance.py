from abc import ABC, abstractmethod

from tw_management_module import tw_offering as two, tw_selection as tws
from core_module import instance as it, solution as slt, tour as tr
from utility_module import utils as ut


class RequestAcceptanceAttractiveness(ABC):
    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def evaluate(self, instance: it.MDPDPTWInstance, carrier: slt.AHDSolution, request: int):
        pass


class FirstComeFirstServed(RequestAcceptanceAttractiveness):
    def evaluate(self, instance: it.MDPDPTWInstance, carrier: slt.AHDSolution, request: int):
        return True


class Dummy(RequestAcceptanceAttractiveness):
    def evaluate(self, instance: it.MDPDPTWInstance, carrier: slt.AHDSolution, request: int):
        return False


class _CloseToCompetitors(RequestAcceptanceAttractiveness):
    """
    Attractiveness is based on distance to competitors. If the distance to the competitor is at least as small as
    the distance to the own depot, a request is rated attractive.
    """

    def evaluate(self, instance: it.MDPDPTWInstance, carrier: slt.AHDSolution, request: int):
        pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(request)
        carrier_dist_sum = instance.distance([pickup_vertex, carrier.id_, carrier.id_],
                                             [delivery_vertex, pickup_vertex, delivery_vertex])
        for competitor_id in range(instance.num_carriers):
            if competitor_id == carrier.id_:
                continue
            competitor_dist_sum = instance.distance([pickup_vertex, competitor_id, competitor_id],
                                                    [delivery_vertex, pickup_vertex, delivery_vertex])
            # self.closeness = 0.5 -> the request is at least as close to the competitor as it si to the carrier
            if competitor_dist_sum / (competitor_dist_sum + carrier_dist_sum) <= self.closeness:
                return True
        return False


class CloseToCompetitors25(_CloseToCompetitors):
    def __init__(self):
        super().__init__()
        self.closeness = .25


class CloseToCompetitors50(_CloseToCompetitors):
    def __init__(self):
        super().__init__()
        self.closeness = .5


class CloseToCompetitors75(_CloseToCompetitors):
    def __init__(self):
        super().__init__()
        self.closeness = .75


class RequestAcceptanceBehavior:
    def __init__(self,
                 max_num_accepted_infeasible: int,
                 request_acceptance_attractiveness: RequestAcceptanceAttractiveness,
                 time_window_offering: two.TWOfferingBehavior,
                 time_window_selection: tws.TWSelectionBehavior,
                 ):
        self.max_num_accepted_infeasible = max_num_accepted_infeasible
        self.request_acceptance_attractiveness = request_acceptance_attractiveness
        self.time_window_offering = time_window_offering
        self.time_window_selection = time_window_selection
        self.name = self.__class__.__name__

    def execute(self, instance: it.MDPDPTWInstance, carrier: slt.AHDSolution, request: int):

        offer_set = self.time_window_offering.execute(instance, carrier, request)
        acceptance_type = 'accept_feasible'
        if not offer_set:
            if not len(carrier.accepted_infeasible_requests) < self.max_num_accepted_infeasible:
                return 'reject_no_pendulum_capacity', None
            elif not self.request_acceptance_attractiveness.evaluate(instance, carrier, request):
                return 'reject_not_attractive', None
            else:
                # override the (currently empty) offer set and the default acceptance type
                offer_set = self.pendulum_feasible_set(instance, carrier, request)
                acceptance_type = 'accept_infeasible'

        selected_tw = self.time_window_selection.select_tw(offer_set, request)
        if not selected_tw:
            return 'reject_preference_mismatch', None
        else:
            return acceptance_type, selected_tw

    """
        # [1] would the request be accepted even if it was infeasible?
        pendulum_capacity_available = len(carrier.accepted_infeasible_requests) < self.max_num_accepted_infeasible
        request_is_attractive = self.request_acceptance_attractiveness.evaluate(instance, carrier, request)
        if pendulum_capacity_available and request_is_attractive:
            accept = True
        else:
            accept = False

        # [2] define offer set. necessary in all cases to determine the acceptance_type
        offer_set = self.time_window_offering.execute(instance, carrier, request)

        # [3] offer all time windows that are feasible for a pendulum tour if it is an attractive request and can be
        # accepted even if its infeasible
        if accept:
            pendulum_feasible_set = self.pendulum_feasible_set(instance, carrier, request)
            assert pendulum_feasible_set, f'Request {request} of carrier {carrier.id_} cannot even be served with a ' \
                                          f'pendulum tour! '
            selected_tw = self.time_window_selection.select_tw(pendulum_feasible_set, request)
            if selected_tw is False:
                acceptance_type = 'reject_no_preference_match'
            elif selected_tw in offer_set:
                acceptance_type = 'accept_feasible'
            else:
                acceptance_type = 'accept_infeasible'

        # [4] offer only the offer_set if request is not attractive or no more pendulum tours are allowed
        else:
            if not offer_set:
                selected_tw = None
                acceptance_type = 'reject_no_offer_set'
            else:
                selected_tw = self.time_window_selection.select_tw(offer_set, request)
                if selected_tw:
                    acceptance_type = 'accept_feasible'
                else:
                    acceptance_type = 'reject_no_preference_match'


        return acceptance_type, selected_tw
    """

    def pendulum_feasible_set(self, instance: it.MDPDPTWInstance, carrier: slt.AHDSolution, request: int):
        """
        returns all time windows for which a pendulum tour is feasible
        """
        pendulum_feasible_set = []
        tour = tr.Tour('temp', carrier.id_)
        pickup, delivery = instance.pickup_delivery_pair(request)
        instance.assign_time_window(pickup, ut.EXECUTION_TIME_HORIZON)
        for tw in ut.ALL_TW:
            instance.assign_time_window(delivery, tw)
            if tour.insertion_feasibility_check(instance, [1, 2], [pickup, delivery]):
                pendulum_feasible_set.append(tw)
        instance.assign_time_window(pickup, ut.EXECUTION_TIME_HORIZON)
        instance.assign_time_window(delivery, ut.EXECUTION_TIME_HORIZON)
        return pendulum_feasible_set
