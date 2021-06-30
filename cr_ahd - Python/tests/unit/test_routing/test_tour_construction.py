import pytest
from math import sqrt
from src.cr_ahd.utility_module import plotting as pl
from src.cr_ahd.routing_module import tour_construction as cns


def test__tour_cheapest_insertion(inst_and_sol_gh_0_ass9_routed6):
    instance, solution = inst_and_sol_gh_0_ass9_routed6
    best_delta = -sqrt(25 ** 2 + 50 ** 2) + 75 - sqrt(25 ** 2 + 100 ** 2) + 125
    best_pickup_pos = 1
    best_deliver_pos = 6
    expected = (best_delta, best_pickup_pos, best_deliver_pos)
    # pytest.parametrize didn't work because the fixtures where not recreated from scratch...
    assert cns.CheapestPDPInsertion()._tour_cheapest_dist_insertion(instance, solution, 0, 0, 2) == expected
    assert cns.CheapestPDPInsertion()._tour_cheapest_dist_insertion(instance, solution, 1, 0, 7) == expected
    assert cns.CheapestPDPInsertion()._tour_cheapest_dist_insertion(instance, solution, 2, 0, 12) == expected

    assert cns.CheapestPDPInsertion()._carrier_insertion_construction(instance, solution, 0, solution.carriers[
        0].unrouted_requests) == (2, 0, 1, 6)

    cns.CheapestPDPInsertion()._execute_insertion(instance, solution, 0, 2, 0, 1, 6)
    assert solution.carriers[0].tours[0].routing_sequence == (0, 5, 4, 19, 3, 18, 20, 0)


def test__tour_cheapest_insertion_2(inst_and_sol_gh_0_9_assigned):
    instance, solution = inst_and_sol_gh_0_9_assigned
    pdp_insertion = cns.CheapestPDPInsertion()
    assert pdp_insertion._carrier_insertion_construction(instance, solution, 0, [0, 1, 2]) == (0, None, None, None)
    pdp_insertion._create_new_tour_with_request(instance, solution, 0, 0)
    # enforce insertion of request 2
    assert pdp_insertion._carrier_insertion_construction(instance, solution, 0, [2]) == (2, 0, 1, 4)
    pdp_insertion._execute_insertion(instance, solution, 0, 2, 0, 2, 3)
    assert pdp_insertion._carrier_insertion_construction(instance, solution, 0,
                                                         solution.carriers[0].unrouted_requests) == (1, 0, 2, 5)
