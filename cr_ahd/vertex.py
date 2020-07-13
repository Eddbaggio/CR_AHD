from utils import TimeWindow, Coords


class Vertex(object):
    """docstring for Request"""

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

    def __str__(self):
        return f'Vertex (ID={self.id_}, {self.coords}, {self.tw}, Demand={self.demand})'


if __name__ == '__main__':
    req1 = Vertex()
