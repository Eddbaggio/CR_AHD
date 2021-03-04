import numpy as np

from src.cr_ahd.utility_module.utils import TimeWindow, Coords, opts
from abc import ABC, abstractmethod


class BaseVertex(ABC):
    def __init__(self, id_: str, x_coord: float, y_coord: float):
        self.id_ = id_  # unique identifier TODO: assert that the id is unique?
        self.coords = Coords(x_coord, y_coord)  # Location in a 2D plane

    @abstractmethod
    def to_dict(self):
        pass


class DepotVertex(BaseVertex):
    def __init__(self, id_: str,
                 x_coord: float,
                 y_coord: float,
                 carrier_assignment: str = None):
        super().__init__(id_, x_coord, y_coord)
        self.carrier_assignment = carrier_assignment
        self.assigned = True  # TODO why is this true by default?
        self.tw = TimeWindow(opts['start_time'], np.infty)
        self.service_duration = 0
        self.demand = 0

    def __str__(self):
        return f'{self.__dict__}'

    def to_dict(self):
        return {
            'id_': self.id_,
            'x_coord': self.coords.x,
            'y_coord': self.coords.y,
            'carrier_assignment': self.carrier_assignment,
        }


class Vertex(BaseVertex):

    def __init__(self,
                 id_: str,
                 x_coord: float,
                 y_coord: float,
                 demand: float,
                 tw_open: float,
                 tw_close: float,
                 service_duration: float = 0,
                 carrier_assignment: str = None,
                 **kwargs
                 ):
        super().__init__(id_, x_coord, y_coord)
        self.demand = demand
        self.tw = TimeWindow(tw_open, tw_close)  # time windows opening and closing
        self.service_duration = service_duration
        self._carrier_assignment = carrier_assignment
        self._assigned = False
        self._routed = False

    def __str__(self):
        return f'Vertex (ID={self.id_}, {self.coords}, {self.tw}, Demand={self.demand}, Carrier={self.carrier_assignment}, routed={self.routed}, assigned={self.assigned})'

    def to_dict(self):
        return {
            'id_': self.id_,
            'x_coord': self.coords.x,
            'y_coord': self.coords.y,
            'demand': self.demand,
            'tw_open': self.tw.e,
            'tw_close': self.tw.l,
            'service_duration': self.service_duration,
            'carrier_assignment': self.carrier_assignment,
            'assigned': self.assigned,
        }

    @property
    def carrier_assignment(self):
        return self._carrier_assignment

    @carrier_assignment.setter
    def carrier_assignment(self, carrier_id):
        assert not self.assigned, f'vertex {self} has already been assigned to carrier {self.carrier_assignment}! Must' \
                                  f'retract before re-assigning'
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
        assert routed ^ self.routed, f'routed attribute of {self} can only set from True to False or vice versa'
        self._routed = routed

# if __name__ == '__main__':
# carrier = carrier.Carrier(-99, Vertex('dummy', 0, 0, 0, 0, 0), [])
# print(carrier)
