from src.cr_ahd.core_module.tour import Tour
from src.cr_ahd.utility_module.utils import opts


class Vehicle(object):
    """docstring for Vehicle"""

    def __init__(self, id_: str, load_capacity, distance_capacity):
        self.id_ = id_
        self.load_capacity = load_capacity
        self.distance_capacity = distance_capacity
        self.tour: Tour = None
        self.color = next(opts['ccycler'])['color']

    def __str__(self):
        return f'Vehicle (ID={self.id_}, capacity={self.load_capacity})'

    @property
    def is_active(self):
        return True if len(self.tour) > 2 else False

    def to_dict(self):
        return {
            'id_': self.id_,
            'capacity': self.load_capacity
        }
