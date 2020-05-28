import request as rq


class Carrier(object):
    """docstring for Carrier"""

    def __init__(self, id_, depot, vehicles):
        self.id_ = id_
        self.depot = depot
        self.vehicles = vehicles
        self.requests = []
        pass

    def __str__(self):
        return f'Carrier (ID:{self.id_}, Depot:{self.depot}, Vehicles:{len(self.vehicles)}, Requests:{len(self.requests)})'

    def assign_request(self, requests):
        if type(requests) == list:
            self.requests.extend(requests)
        elif type(requests) == rq.Request:
            self.requests.append(requests)
        pass

    def function(self):
        pass
