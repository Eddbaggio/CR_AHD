import time
from functools import wraps
from typing import Callable

import src.cr_ahd.core_module.solution as slt


def timing(f: Callable):
    @wraps(f)
    def wrap(*args, **kw):
        start_time = time.perf_counter()
        result = f(*args, **kw)
        end_time = time.perf_counter()
        # print(f'func:{f.__name__} args:[{args}, {kw}] took: {end_time - start_time} sec')

        # store the timing in the solution
        # only works if the function had a slt.CAHDSolution as an input ...
        if isinstance(result, slt.CAHDSolution):
            try:
                result.timings[f.__name__] += [end_time - start_time]
            except KeyError:
                result.timings[f.__name__] = [end_time - start_time]
        else:
            for arg in [*args, *kw.values()]:
                if isinstance(arg, slt.CAHDSolution):
                    try:
                        arg.timings[f.__name__] += [end_time - start_time]
                    except KeyError:
                        arg.timings[f.__name__] = [end_time - start_time]
        return result

    return wrap


class Timer(object):
    def __init__(self):
        self._start_set = False
        self._stop_set = False
        self._paused = False
        self._pause_duration = 0

    def start(self):
        assert not self._start_set
        self._start_set = True
        self._start = time.perf_counter()

    def stop(self):
        assert not self._stop_set
        self._stop_set = True
        self._end = time.perf_counter()

    def pause(self):
        assert not self._paused
        self._pause_start = time.perf_counter()
        self._paused = True

    def resume(self):
        assert self._paused
        self._pause_end = time.perf_counter()
        self._pause_duration += self._pause_end - self._pause_start
        self._paused = False

    @property
    def duration(self):
        assert self._stop_set
        return self._end - self._start - self._pause_duration
