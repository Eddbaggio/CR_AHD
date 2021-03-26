from src.cr_ahd.utility_module import utils
import pandas as pd


class Test:
    def test_travel_time(self):
        assert utils.travel_time(50) == (50 / utils.opts['speed_kmh']) * 60 ** 2

    def test_random_max_k_partition(self):
        """cannot actually be tested since it's random"""
        ls = range(10)
        k = 4
        partition = utils.random_max_k_partition(ls, k)
        assert len(partition) <= k

    def test_make_travel_time_matrix(self, request_vertices_f):
        tt_matrix = utils.make_travel_time_matrix(request_vertices_f)
        assert tt_matrix == pd.DataFrame(

        )
