import math

import pytest
from src.cr_ahd.core_module import instance as it


class TestPDPInstance:

    def test_num_carriers(self, inst_gh_0: it.PDPInstance):
        assert inst_gh_0.num_carriers == 3

    def test_num_requests(self, inst_gh_0: it.PDPInstance):
        assert inst_gh_0.num_requests == 15

    def test_distance(self, inst_gh_0: it.PDPInstance):
        """
        somewhat arbitrary points checked for the correct distance
        """
        inst = inst_gh_0
        # distance between depots
        assert pytest.approx(inst.distance([0], [1]), abs=0.2) == 200
        assert pytest.approx(inst.distance([0], [2]), abs=0.2) == 200
        assert pytest.approx(inst.distance([1], [2]), abs=0.2) == 200

        # some random points
        assert inst.distance([3], [18]) == 50
        assert inst.distance([18], [3]) == 50

        assert inst.distance([0], [5]) == 25
        assert inst.distance([5], [0]) == 25
        assert pytest.approx(inst.distance([0], [22])) == math.sqrt((25 ** 2 + 100 ** 2))

        assert inst.distance([3], [8]) == 150
        # first and last index
        assert inst.distance([0], [inst.num_carriers + inst.num_requests * 2 - 1]) == math.sqrt(75 ** 2 + 273 ** 2)

        pass

    @pytest.mark.parametrize("test_input, expected", [
        (0, (3, 18)),
        (14, (17, 32)),
    ])
    def test_pickup_delivery_pair(self, test_input, expected, inst_gh_0: it.PDPInstance):
        assert inst_gh_0.pickup_delivery_pair(test_input) == expected

    def test_pickup_delivery_pair_exception(self, inst_gh_0: it.PDPInstance):
        with pytest.raises(IndexError):
            inst_gh_0.pickup_delivery_pair(15)

    @pytest.mark.parametrize("test_input, expected", [
        (3, 0),
        (32, 14),
    ])
    def test_request_from_vertex(self, inst_gh_0: it.PDPInstance, test_input, expected):
        assert inst_gh_0.request_from_vertex(test_input) == expected

    @pytest.mark.parametrize("test_input", [0, 2, 33])
    def test_request_from_vertex_exception(self, inst_gh_0: it.PDPInstance, test_input):
        with pytest.raises(IndexError):
            inst_gh_0.request_from_vertex(test_input)
