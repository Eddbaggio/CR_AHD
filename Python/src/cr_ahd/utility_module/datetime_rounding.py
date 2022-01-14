import datetime as dt


def round_timedelta(td: dt.timedelta, resolution: str):
    """

    :param td: timedelta value
    :param resolution: 's' for seconds, 'm' for minutes, 'h' for hours'
    :return:
    """
    assert td >= dt.timedelta(0), f'Not implemented for negative values'
    if resolution == 's':
        res = dt.timedelta(seconds=round(td.total_seconds(), None))
    elif resolution == 'm':
        res = dt.timedelta(minutes=(int(td.total_seconds() // 60)))
    elif resolution == 'h':
        res = dt.timedelta(minutes=(int(td.total_seconds() // 60 ** 2)))
    else:
        raise ValueError()
    return res


def ceil_timedelta(td: dt.timedelta, resolution: str):
    """

    :param td:
    :param resolution: 's' for seconds, 'm' for minutes, 'h' for hours'
    :return:
    """
    assert td >= dt.timedelta(0), f'Not implemented for negative values'
    a = td.total_seconds()
    if resolution == 's':
        b = 1
        x = a // b + bool(a % b)
        res = dt.timedelta(seconds=x)
    elif resolution == 'm':
        b = 60
        x = a // b + bool(a % b)
        res = dt.timedelta(minutes=x)
    elif resolution == 'h':
        b = 60 ** 2
        x = a // b + bool(a % b)
        res = dt.timedelta(hours=x)
    else:
        raise ValueError()

    return res
