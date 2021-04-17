import datetime as dt

import numpy as np
import pytest
import random
from pathlib import Path
import pandas as pd

import src.cr_ahd.utility_module.utils as ut
from src.cr_ahd.core_module import instance as it, solution as sl, tour as tr


@pytest.fixture
def gansterer_hartl_3C_MV_0():
    return it.read_gansterer_hartl_mv(Path('../fixtures/test=1+dist=200+rad=200+n=5.dat'))


@pytest.fixture
def instance_0():
    return it.PDPInstance('fixture_instance_0',
                          requests=pd.DataFrame({

                          }))

@pytest.fixture
def tour_0(gansterer_hartl_3C_MV_0):
    instance = gansterer_hartl_3C_MV_0()
    solution = sl.GlobalSolution(instance)
    return tr.Tour('fixture_tour_0', instance, solution, 0)

@pytest.fixture
def request_vertices_spiral():
    """https://stackoverflow.com/questions/398299/looping-in-a-spiral"""

    def _request_vertices_spiral(X, Y):
        requests = []
        x = y = 0
        dx = 0
        dy = -1
        for i in range(max(X, Y) ** 2):
            if (-X / 2 < x <= X / 2) and (-Y / 2 < y <= Y / 2):
                requests.append(
                    Vertex(f'r{i - 1}', x, y, 0, ut.START_TIME + i * ut.TW_LENGTH,
                           ut.START_TIME + (i + 1) * ut.TW_LENGTH, dt.timedelta(minutes=10)))
            if x == y or (x < 0 and x == -y) or (x > 0 and x == 1 - y):
                dx, dy = -dy, dx
            x, y = x + dx, y + dy
        return requests[1:]

    return _request_vertices_spiral
