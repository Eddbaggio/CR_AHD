import pytest

import src.cr_ahd.utility_module.utils as ut
import pandas as pd
import datetime as dt
import math


class Test:
    def test_travel_time(self):
        assert ut.travel_time(50) == dt.timedelta(hours=50 / ut.opts['speed_kmh'])

    def test_random_max_k_partition(self):
        """cannot actually be tested since it's random"""
        ls = range(10)
        k = 4
        partition = ut.random_max_k_partition(ls, k)
        assert len(partition) <= k

    def test_make_travel_dist_matrix(self, request_vertices_b):
        td_matrix = ut.make_travel_dist_matrix(request_vertices_b)
        result = {
            'r0': [0, math.sqrt(200), 20, math.sqrt(200)],
            'r1': [math.sqrt(200), 0, math.sqrt(200), 20],
            'r2': [20, math.sqrt(200), 0, math.sqrt(200)],
            'r3': [math.sqrt(200), 20, math.sqrt(200), 0],
        }
        pd.testing.assert_frame_equal(td_matrix, pd.DataFrame(data=result, index=['r0', 'r1', 'r2', 'r3']))

    def test_make_travel_time_matrix(self, request_vertices_f):
        tt_matrix = ut.make_travel_duration_matrix(request_vertices_f)
        speed = ut.opts['speed_kmh']  # km/h
        result = {
            'd0': [dt.timedelta(0), dt.timedelta(hours=10/speed), dt.timedelta(hours=math.sqrt(200)/speed), dt.timedelta(hours=math.sqrt(200)/speed), dt.timedelta(hours=math.sqrt(200) / speed),  dt.timedelta(hours=20/speed)],
            'r0': [dt.timedelta(hours=10/speed), dt.timedelta(0), dt.timedelta(hours=10/speed), dt.timedelta(hours=math.sqrt(500)/speed), dt.timedelta(hours=math.sqrt(500)/speed), dt.timedelta(hours=math.sqrt(500)/speed)],
            'r1': [dt.timedelta(hours=math.sqrt(200)/speed), dt.timedelta(hours=10/speed), dt.timedelta(0), dt.timedelta(hours=20/speed), dt.timedelta(hours=math.sqrt(800)/speed), dt.timedelta(hours=math.sqrt(1000)/speed)],
            'r2': [dt.timedelta(hours=math.sqrt(200)/speed), dt.timedelta(hours=math.sqrt(500)/speed), dt.timedelta(hours=20/speed), dt.timedelta(0), dt.timedelta(hours=20/speed), dt.timedelta(hours=math.sqrt(1000)/speed)],
            'r3': [dt.timedelta(hours=math.sqrt(200)/speed), dt.timedelta(hours=math.sqrt(500)/speed), dt.timedelta(hours=math.sqrt(800)/speed), dt.timedelta(hours=20/speed), dt.timedelta(0), dt.timedelta(hours=math.sqrt(200)/speed)],
            'r4': [dt.timedelta(hours=20/speed), dt.timedelta(hours=math.sqrt(500)/speed), dt.timedelta(hours=math.sqrt(1000)/speed), dt.timedelta(hours=math.sqrt(1000)/speed), dt.timedelta(hours=math.sqrt(200)/speed), dt.timedelta(0)],
        }
        result = pd.DataFrame(data=result, index=['d0', 'r0', 'r1', 'r2', 'r3', 'r4'])
        pd.testing.assert_frame_equal(tt_matrix, result, check_dtype=False)
