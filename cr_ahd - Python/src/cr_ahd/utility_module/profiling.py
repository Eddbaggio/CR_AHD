import time
import inspect
from functools import wraps

import src.cr_ahd.core_module.solution as slt


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        # print(f'func:{f.__name__} args:[{args}, {kw}] took: {te - ts} sec')

        # store the timing in the solution
        # only works if the function had a slt.CAHDSolution as an input ...
        if isinstance(result, slt.CAHDSolution):
            try:
                result.timings[f.__name__] += [te - ts]
            except KeyError:
                result.timings[f.__name__] = [te - ts]
        else:
            for arg in [*args, *kw.values()]:
                if isinstance(arg, slt.CAHDSolution):
                    try:
                        arg.timings[f.__name__] += [te - ts]
                    except KeyError:
                        arg.timings[f.__name__] = [te - ts]
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
