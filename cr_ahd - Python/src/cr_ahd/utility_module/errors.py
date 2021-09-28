class InsertionError(Exception):
    """Exception raised for errors in the insertion of a request into a tour.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class ConstraintViolationError(Exception):
    def __init__(self, expression='', message=''):
        self.expression = expression
        self.message = message