import abc
import datetime as dt

from core_module import instance as it, solution as slt, tour as tr
from tw_management_module import tw
from utility_module import utils as ut


class TWOfferingBehavior(abc.ABC):
    def __init__(self, time_window_length: dt.timedelta):
        self.name = self.__class__.__name__
        self.time_window_length = time_window_length
        self.time_windows = ut.generate_time_windows(time_window_length)

    def execute(self, instance: it.CAHDInstance, carrier: slt.AHDSolution, request: int):
        delivery_vertex = instance.vertex_from_request(request)
        # make sure that the request has not been given a tw yet
        assert instance.tw_open[delivery_vertex] in (ut.EXECUTION_START_TIME, None), \
            f'tw_open={instance.tw_open[delivery_vertex]}'
        assert instance.tw_close[delivery_vertex] in (ut.END_TIME, None), \
            f'tw_close={instance.tw_close[delivery_vertex]}'

        tw_valuations = []
        for tw in self.time_windows:
            tw_valuations.append(self._evaluate_time_window(instance, carrier, request, tw))
        offered_time_windows = list(sorted(zip(tw_valuations, self.time_windows), key=lambda x: x[0]))
        offered_time_windows = [tw for valuation, tw in offered_time_windows if valuation >= 0]
        return offered_time_windows

    @abc.abstractmethod
    def _evaluate_time_window(self, instance: it.CAHDInstance, carrier: slt.AHDSolution, request: int,
                              tw: tw.TimeWindow):
        pass


class FeasibleTW(TWOfferingBehavior):
    def _evaluate_time_window(self, instance: it.CAHDInstance, carrier: slt.AHDSolution, request: int,
                              tw: tw.TimeWindow):
        """
        :return: 1 if TW is feasible, -1 else
        """

        delivery_vertex = instance.vertex_from_request(request)
        # temporarily set the time window under consideration
        instance.tw_open[delivery_vertex] = tw.open
        instance.tw_close[delivery_vertex] = tw.close

        # can the carrier open a new pendulum tour and insert the request there?
        if len(carrier.tours) < instance.carriers_max_num_tours:

            tmp_tour = tr.Tour('tmp', carrier.id_)
            if tmp_tour.insertion_feasibility_check(instance, [1], [delivery_vertex]):
                # undo the setting of the time window and return
                instance.tw_open[delivery_vertex] = ut.EXECUTION_START_TIME
                instance.tw_close[delivery_vertex] = ut.END_TIME
                return 1

        # if no feasible new tour can be built, can the request be inserted into one of the existing tours?
        for tour in carrier.tours:
            for delivery_pos in range(1, len(tour)):
                if tour.insertion_feasibility_check(instance, [delivery_pos], [delivery_vertex]):
                    # undo the setting of the time window and return
                    instance.tw_open[delivery_vertex] = ut.EXECUTION_START_TIME
                    instance.tw_close[delivery_vertex] = ut.END_TIME
                    return 1

        # undo the setting of the time window and return
        instance.tw_open[delivery_vertex] = ut.EXECUTION_START_TIME
        instance.tw_close[delivery_vertex] = ut.END_TIME
        return -1


class NoTw(TWOfferingBehavior):
    def execute(self, instance: it.CAHDInstance, carrier: slt.AHDSolution, request: int):
        return [ut.EXECUTION_TIME_HORIZON]

    def _evaluate_time_window(self, instance: it.CAHDInstance, carrier: slt.AHDSolution, request: int,
                              tw: tw.TimeWindow):
        pass
