import pytest
from math import sqrt

from src.cr_ahd.routing_module import tour_construction as cns


class TestInsertionConstruction:
    def test__tour_cheapest_insertion(self, inst_and_sol_gh_0_9_ass_6_routed):
        instance, solution = inst_and_sol_gh_0_9_ass_6_routed
        best_delta = -sqrt(25 ** 2+50**2) + 75 - sqrt(25 ** 2 + 100 ** 2) + 125
        best_pickup_pos = 1
        best_deliver_pos = 6
        expected = (best_delta, best_pickup_pos, best_deliver_pos)
        # pytest.parametrize didn't work because the fixtures where not recreated from scratch...
        assert cns.CheapestInsertion()._tour_cheapest_insertion(instance, solution, 0, 0, 2) == expected
        assert cns.CheapestInsertion()._tour_cheapest_insertion(instance, solution, 1, 0, 7) == expected
        assert cns.CheapestInsertion()._tour_cheapest_insertion(instance, solution, 2, 0, 12) == expected
