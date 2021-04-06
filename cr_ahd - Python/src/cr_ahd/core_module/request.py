import abc
import datetime as dt
from typing import Tuple

from src.cr_ahd.core_module import vertex as vx
from src.cr_ahd.utility_module import utils as ut


class Request(abc.ABC):
    def __init__(self, id_, carrier_assignment, revenue: float, load: float, service_duration: dt.timedelta):
        self.id_ = id_
        self._carrier_assignment = carrier_assignment
        self._revenue = revenue
        self._load = load
        self._service_duration = service_duration
        self._assigned = False
        self._routed = False

    @property
    def revenue(self):
        return self._revenue

    @property
    def load(self):
        return self._load

    @property
    def carrier_assignment(self):
        return self._carrier_assignment

    @carrier_assignment.setter
    def carrier_assignment(self, carrier_id):
        assert not self.assigned, f'request {self} has already been assigned to carrier {self.carrier_assignment}! ' \
                                  f'Must retract before re-assigning'
        self._carrier_assignment = carrier_id

    @property
    def assigned(self):
        return self._assigned

    @assigned.setter
    def assigned(self, assigned: bool):
        # XOR can only set from true to false or vice versa
        assert self.assigned ^ assigned
        self._assigned = assigned

    @property
    def routed(self):
        return self._routed

    @routed.setter
    def routed(self, routed: bool):
        # XOR: can only set from True to False or vice versa
        assert bool(routed) ^ bool(
            self.routed), f'routed attribute of {self} can only set from True to False or vice versa'
        self._routed = routed

    @abc.abstractmethod
    def __str__(self):
        pass

    @property
    @abc.abstractmethod
    def vertices(self) -> Tuple:
        pass

    @abc.abstractmethod
    def to_dict(self):
        pass


class DeliveryRequest(Request):
    """Request that only comprises a delivery location but no pickup."""

    def __init__(self, id_, x, y,
                 tw_open: dt.datetime, tw_close: dt.datetime,
                 carrier_assignment,
                 revenue: float, load: float,
                 service_duration: dt.timedelta = dt.timedelta(0)):
        super().__init__(id_, carrier_assignment, revenue, load, service_duration)
        self.delivery_vertex = vx.Vertex(id_, x, y, tw_open, tw_close)

    @property
    def vertices(self):
        return self.delivery_vertex,

    def to_dict(self):
        pass

    def __str__(self):
        return f'{self.id_}: delivery={self.delivery_vertex.coords},' \
               f' carrier_assignment={self.carrier_assignment}, revenue={self.revenue}, load={self.load}'


class PickupAndDeliveryRequest(Request):
    def __init__(self, id_, pickup_x: float, pickup_y: float, delivery_x: float, delivery_y: float,
                 delivery_tw_open: dt.datetime, delivery_tw_close: dt.datetime,
                 carrier_assignment,
                 revenue: float, load: float,
                 service_duration: dt.timedelta = dt.timedelta(0), pickup_tw_open=ut.START_TIME,
                 pickup_tw_close=ut.END_TIME):
        super().__init__(id_, carrier_assignment, revenue, load, service_duration)
        self.pickup_vertex = vx.Vertex(f'p_{id_}', pickup_x, pickup_y, pickup_tw_open, pickup_tw_close)
        self.delivery_vertex = vx.Vertex(f'd_{id_}', delivery_x, delivery_y, delivery_tw_open, delivery_tw_close)

    @property
    def vertices(self):
        return self.pickup_vertex, self.delivery_vertex

    def to_dict(self):
        pass

    def __str__(self):
        return f'{self.id_}: ' \
               f'pickup={self.pickup_vertex.coords}, delivery={self.delivery_vertex.coords}, ' \
               f'delivery_tw={self.delivery_vertex.tw}, ' \
               f'carrier_assignment={self.carrier_assignment}, ' \
               f'revenue={self.revenue}, load={self.load}'
