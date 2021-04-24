import pytest
from src.cr_ahd.routing_module import tour_improvement as imp
from src.cr_ahd.utility_module import plotting as pl


def test_PDPTwoOptBest(inst_and_sol_gh_0_9_ass_6_routed):
    instance, solution = inst_and_sol_gh_0_9_ass_6_routed
    carrier = 0
    tour = 0
    local_search = imp.PDPTwoOptBest()
    move = local_search.improve_tour(instance, solution, carrier, tour)
    best_delta, best_pos_i, best_pos_j = move
    assert best_delta == 0
    assert best_pos_i is None
    assert best_pos_j is None

