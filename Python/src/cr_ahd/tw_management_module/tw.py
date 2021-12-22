import datetime as dt


class TimeWindow:
    def __init__(self, open: dt.datetime, close: dt.datetime):
        self.open: dt.datetime = open
        self.close: dt.datetime = close

    def __str__(self):
        return f'[D{self.open.day} {self.open.strftime("%H:%M:%S")} - D{self.close.day} {self.close.strftime("%H:%M:%S")}]'

    def __repr__(self):
        return f'[D{self.open.day} {self.open.strftime("%H:%M:%S")} - D{self.close.day} {self.close.strftime("%H:%M:%S")}]'