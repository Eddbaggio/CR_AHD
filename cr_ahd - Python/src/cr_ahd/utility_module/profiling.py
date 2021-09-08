import time
from functools import wraps
from typing import Callable

import src.cr_ahd.core_module.solution as slt

'''
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
'''


class Timer(object):
    def __init__(self, start=True):
        self._started = False
        self._start_time = None
        self._stopped = False
        self._stop_time = None
        self._paused = False
        self._pause_start_time = None
        self._pause_end_time = None
        self._pause_duration = 0

        if start is True:
            self.start()

    def start(self):
        assert not self._started
        self._started = True
        self._start_time = time.perf_counter()

    def stop(self):
        assert not self._stopped
        self._stopped = True
        self._stop_time = time.perf_counter()

    def pause(self):
        assert not self._paused
        self._pause_start_time = time.perf_counter()
        self._paused = True

    def resume(self):
        assert self._paused
        self._pause_end_time = time.perf_counter()
        self._pause_duration += self._pause_end_time - self._pause_start_time
        self._paused = False

    @property
    def duration(self):
        assert self._stopped
        return self._stop_time - self._start_time - self._pause_duration

    def write_duration_to_solution(self, solution: slt.CAHDSolution, name: str, add_to_existing: bool = False):
        """
        Stops the timer if it has not been stopped yet. Then, writes the timer's measured duration to the solution
        with the given name. If there already exists a measurement under the given name, add_to_existing determines
        whether the operation will fail or the time will be added to the existing one

        :param solution:
        :param name:
        :return:
        """
        if self._stopped is False:
            self.stop()
        if name in solution.timings:
            if add_to_existing:
                solution.timings[name] += self.duration
            else:
                raise KeyError(f'Name {name} already exists but add_to_existing was set to {add_to_existing}')
        else:
            solution.timings[name] = self.duration
