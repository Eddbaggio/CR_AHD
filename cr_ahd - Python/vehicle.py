from tour import Tour
from utils import opts


class Vehicle(object):
    """docstring for Vehicle"""

    def __init__(self, id_: str, capacity):
        self.id_ = id_
        self.capacity = capacity
        self.tour: Tour = None
        self.color = next(opts['ccycler'])['color']

    def __str__(self):
        return f'Vehicle (ID={self.id_}, capacity={self.capacity})'

    @property
    def is_active(self):
        return True if len(self.tour) > 2 else False

    def to_dict(self):
        return {
            'id_': self.id_,
            'capacity': self.capacity
        }
