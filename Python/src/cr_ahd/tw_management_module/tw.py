import datetime as dt


class TimeWindow:
    def __init__(self, open: dt.datetime, close: dt.datetime):
        assert close >= open
        self.open: dt.datetime = open
        self.close: dt.datetime = close
        self.duration: dt.timedelta = self.close - self.open

    def __str__(self):
        return f'[D{self.open.day} {self.open.strftime("%H:%M:%S")} - ' \
               f'D{self.close.day} {self.close.strftime("%H:%M:%S")}]'

    def __repr__(self):
        return f'[D{self.open.day} {self.open.strftime("%H:%M:%S")} - ' \
               f'D{self.close.day} {self.close.strftime("%H:%M:%S")}]'

    def __eq__(self, other):
        return self.open == other.open and self.close == other.close

    def overlap(self, other):
        other: TimeWindow
        open = max(self.open, other.open)
        close = min(self.close, other.close)
        if close > open:
            return TimeWindow(open, close)
        else:
            return None
