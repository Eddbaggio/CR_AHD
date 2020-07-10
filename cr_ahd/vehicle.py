from utils import opts


class Vehicle(object):
    """docstring for Vehicle"""

    def __init__(self, id_, capacity):
        self.id_ = id_
        self.capacity = capacity
        self.tour = None
        self.color = [next(opts['ccycler'])['color'], ]

    def __str__(self):
        return f'Vehicle (ID={self.id_}, capacity={self.capacity})'
