from utils import TimeWindow, Coords
import carrier


class Vertex(object):

    def __init__(self,
                 id_: str,
                 x_coord: float,
                 y_coord: float,
                 demand: float,
                 tw_open: float,
                 tw_close: float,
                 service_duration: float = 0):

        self.id_ = id_  # unique identifier TODO: assert that the id is unique?
        self.coords = Coords(x_coord, y_coord)  # Location in a 2D plane
        self.demand = demand
        self.tw = TimeWindow(tw_open, tw_close)  # time windows opening and closing
        self.service_duration = service_duration
        self._carrier_assignment = None

    def __str__(self):
        return f'Vertex (ID={self.id_}, {self.coords}, {self.tw}, Demand={self.demand})'

    def to_dict(self):
        return {
            'id_': self.id_,
            'x_coord': self.coords.x,
            'y_coord': self.coords.y,
            'demand': self.demand,
            'tw_open': self.tw.e,
            'tw_close': self.tw.l,
            'service_duration': self.service_duration
        }

    @property
    def carrier_assignment(self):
        return self._carrier_assignment

    @carrier_assignment.setter
    def carrier_assignment(self, carrier_id):
        assert not self._carrier_assignment, f'vertex has already been assigned to carrier {self.carrier_assignment}!'
        self._carrier_assignment = carrier_id




# if __name__ == '__main__':
    # carrier = carrier.Carrier(-99, Vertex('dummy', 0, 0, 0, 0, 0), [])
    # print(carrier)
