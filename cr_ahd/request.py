class Request(object):
    """docstring for Request"""

    def __init__(self, id_, x, y, e, l):
        self.id_ = id_  # unique identifier TODO: assert that the id is unique?
        self.coords = (x, y)  # Location in a 2D plane
        self.tw = (e, l)  # time windows opening and closing

    def __str__(self):
        return f'Request (ID={self.id_}, location={self.coords}, time window={self.tw})'


if __name__ == '__main__':
    req1 = Request()
