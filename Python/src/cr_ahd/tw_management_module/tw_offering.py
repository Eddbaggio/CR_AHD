import abc
import datetime as dt

from tw_management_module import tw
from core_module import instance as it, solution as slt, tour as tr
from utility_module import utils as ut


class TWOfferingBehavior(abc.ABC):
    def __init__(self):
        self.name = self.__class__.__name__

    def execute(self, instance: it.MDPDPTWInstance, carrier: slt.AHDSolution, request: int):
        pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(request)
        # make sure that the request has not been given a tw yet
        assert instance.tw_open[delivery_vertex] in (ut.EXECUTION_START_TIME, None)
        assert instance.tw_close[delivery_vertex] in (ut.END_TIME, None)

        tw_valuations = []
        for tw in ut.ALL_TW:
            tw_valuations.append(self._evaluate_time_window(instance, carrier, request, tw))
        offered_time_windows = list(sorted(zip(tw_valuations, ut.ALL_TW), key=lambda x: x[0]))
        offered_time_windows = [tw for valuation, tw in offered_time_windows if valuation >= 0]
        return offered_time_windows

    @abc.abstractmethod
    def _evaluate_time_window(self, instance: it.MDPDPTWInstance, carrier: slt.AHDSolution, request: int,
                              tw: tw.TimeWindow):
        pass


class FeasibleTW(TWOfferingBehavior):
    def _evaluate_time_window(self, instance: it.MDPDPTWInstance, carrier: slt.AHDSolution, request: int,
                              tw: tw.TimeWindow):
        """
        :return: 1 if TW is feasible, -1 else
        """

        pickup_vertex, delivery_vertex = instance.pickup_delivery_pair(request)
        # temporarily set the time window under consideration
        instance.tw_open[delivery_vertex] = tw.open
        instance.tw_close[delivery_vertex] = tw.close

        # can the carrier open a new pendulum tour and insert the request there?
        if len(carrier.tours) < instance.carriers_max_num_tours:

            tmp_tour = tr.Tour('tmp', carrier.id_)
            if tmp_tour.insertion_feasibility_check(instance, [1, 2], [pickup_vertex, delivery_vertex]):
                # undo the setting of the time window and return
                instance.tw_open[delivery_vertex] = ut.EXECUTION_START_TIME
                instance.tw_close[delivery_vertex] = ut.END_TIME
                return 1

        # if no feasible new tour can be built, can the request be inserted into one of the existing tours?
        for tour in carrier.tours:
            for pickup_pos in range(1, len(tour)):
                for delivery_pos in range(pickup_pos + 1, len(tour) + 1):
                    if tour.insertion_feasibility_check(
                            instance,
                            [pickup_pos, delivery_pos],
                            [pickup_vertex, delivery_vertex]):
                        # undo the setting of the time window and return
                        instance.tw_open[delivery_vertex] = ut.EXECUTION_START_TIME
                        instance.tw_close[delivery_vertex] = ut.END_TIME
                        return 1

        # undo the setting of the time window and return
        instance.tw_open[delivery_vertex] = ut.EXECUTION_START_TIME
        instance.tw_close[delivery_vertex] = ut.END_TIME
        return -1


class NoTw(TWOfferingBehavior):
    def execute(self, instance: it.MDPDPTWInstance, carrier: slt.AHDSolution, request: int):
        return [ut.EXECUTION_TIME_HORIZON]

    def _evaluate_time_window(self, instance: it.MDPDPTWInstance, carrier: slt.AHDSolution, request: int,
                              tw: tw.TimeWindow):
        pass
