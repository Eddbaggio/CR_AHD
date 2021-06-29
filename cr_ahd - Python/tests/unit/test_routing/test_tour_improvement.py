import math

import pytest
from src.cr_ahd.routing_module import tour_improvement as imp
from src.cr_ahd.utility_module import plotting as pl


def test_PDPTwoOptBest(inst_and_sol_gh_0_ass9_routed6):
    instance, solution = inst_and_sol_gh_0_ass9_routed6
    carrier = 0
    tour = 0
    local_search = imp.PDPTwoOpt()
    move = local_search.find_feasible_move(instance, solution, carrier, tour, False)
    best_delta, best_pos_i, best_pos_j = move
    assert best_delta == 0
    assert best_pos_i is None
    assert best_pos_j is None


def test_PDPMoveBest(inst_and_sol_gh_0_ass9_routed6):
    instance, solution = inst_and_sol_gh_0_ass9_routed6
    # pl.plot_solution_2(instance, solution, title="Before PDPMoveBest", show=True)
    carrier = 0
    tour = 0
    local_search = imp.PDPMove()
    move = local_search.find_feasible_move(instance, solution, carrier, tour, False)
    delta, old_pickup_pos, old_delivery_pos, new_pickup_pos, new_delivery_pos = move
    assert delta == pytest.approx(
        -(50 + math.sqrt(50 ** 2 + 50 ** 2) + math.sqrt(25 ** 2 + 100 ** 2)) + (50 + 50 + math.sqrt(25 ** 2 + 50 ** 2)))
    assert old_pickup_pos == 1
    assert old_delivery_pos == 2
    assert new_pickup_pos == 1
    assert new_delivery_pos == 4
    local_search.execute_move(instance, solution, carrier, tour, move)
    assert solution.carriers[carrier].tours[tour].routing_sequence == (0, 4, 3, 18, 19, 0)
    # pl.plot_solution_2(instance, solution, title=f"After PDPMoveBest; move: {move}", show=True)
