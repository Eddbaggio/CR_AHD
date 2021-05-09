from src.cr_ahd.core_module import instance as it, solution as slt, tour as tr
import pytest
import math


# def test_removal_distance_delta(inst_and_sol_gh_0_ass9_routed6):
#     instance: it.PDPInstance
#     solution: slt.GlobalSolution
#     instance, solution = inst_and_sol_gh_0_ass9_routed6
#     c = 0
#     t = 0
#     tour = solution.carrier_solutions[c].tours[t]
#     delta = tour.removal_distance_delta(instance, [1])
#     assert delta == 50

def test__insert_no_update(inst_and_sol_gh_0_ass9_routed6):
    instance: it.PDPInstance
    solution: slt.GlobalSolution
    instance, solution = inst_and_sol_gh_0_ass9_routed6
    c = 0
    t = 0
    tour = solution.carriers[c].tours[t]
    tour._insert_no_update(1, 5)
    assert tour.routing_sequence == (0, 5, 4, 19, 3, 18, 0)
    # assert tour.revenue_sequence == ()
    # assert tour.load_sequence == ()
    # assert tour.travel_distance_sequence == ()
    # assert tour.travel_duration_sequence == ()


# def test_insertion_distance_delta(inst_and_sol_gh_0_ass9_routed6):
#     instance: it.PDPInstance
#     solution: slt.GlobalSolution
#     instance, solution = inst_and_sol_gh_0_ass9_routed6
#     c = 0
#     t = 0
#     tour = solution.carrier_solutions[c].tours[t]
#     delta = tour.insertion_distance_delta(instance, [1, 6], [5, 20])
#     assert delta == pytest.approx(200 - math.sqrt(25 ** 2 + 50 ** 2) - math.sqrt(100 ** 2 + 25 ** 2))


def test_insert_and_update(inst_and_sol_gh_0_ass9_routed6):
    instance: it.PDPInstance
    solution: slt.GlobalSolution
    instance, solution = inst_and_sol_gh_0_ass9_routed6
    c = 0
    t = 0
    tour = solution.carriers[c].tours[t]
    distance_before = tour.sum_travel_distance
    tour.insert_and_update(instance, solution, [1, 6], [5, 20])
    assert tour.routing_sequence == (0, 5, 4, 19, 3, 18, 20, 0)
    assert tour.sum_travel_distance == pytest.approx(
        distance_before + 200 - math.sqrt(25 ** 2 + 50 ** 2) - math.sqrt(100 ** 2 + 25 ** 2))


def test__pop_no_update(inst_and_sol_gh_0_ass9_routed6):
    instance: it.PDPInstance
    solution: slt.GlobalSolution
    instance, solution = inst_and_sol_gh_0_ass9_routed6
    c = 0
    t = 0
    tour = solution.carriers[c].tours[t]
    tour.pop_no_update(3)
    assert tour.routing_sequence == (0, 4, 19, 18, 0)
    # assert tour.revenue_sequence == ()
    # assert tour.load_sequence == ()
    # assert tour.travel_distance_sequence == ()
    # assert tour.travel_duration_sequence == ()


def test__pop_and_update(inst_and_sol_gh_0_ass9_routed6):
    instance: it.PDPInstance
    solution: slt.GlobalSolution
    instance, solution = inst_and_sol_gh_0_ass9_routed6
    c = 0
    t = 0
    tour = solution.carriers[c].tours[t]
    travel_dist_before = tour.sum_travel_distance
    tour.pop_and_update(instance, solution, [1, 2])
    assert tour.routing_sequence == (0, 3, 18, 0)
    assert tour.sum_travel_distance == pytest.approx(
        travel_dist_before - math.sqrt(25 ** 2 + 50 ** 2) - 50 - math.sqrt(50 ** 2 + 50 ** 2) + math.sqrt(
            25 ** 2 + 100 ** 2))

