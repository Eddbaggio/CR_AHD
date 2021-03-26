from src.cr_ahd.core_module.tour import Tour
from src.cr_ahd.solving_module.tour_improvement import TwoOpt
from src.cr_ahd.utility_module.utils import InsertionError, make_dist_matrix
import pytest
import numpy as np


def test_tour_insert_and_pop(depot_vertex, request_vertices_b):
    distance_matrix = make_dist_matrix([*request_vertices_b, depot_vertex])
    tour = Tour('test', depot_vertex, distance_matrix)
    for request in request_vertices_b:
        tour.insert_and_update(1, request)
    assert all([r.routed for r in request_vertices_b])
    assert tour.sum_travel_durations == 20 + 3 * np.sqrt(200)
    for request in request_vertices_b:
        tour.pop_and_update(request.routed[1])
    assert not any([r.routed for r in request_vertices_b])
    assert tour.sum_travel_durations == 0


def test_tour_cost(small_criss_cross_tour: Tour):
    assert small_criss_cross_tour.sum_travel_durations == 80


def test_TwoOpt_cost(small_criss_cross_tour: Tour):
    TwoOpt().improve_tour(small_criss_cross_tour)
    assert small_criss_cross_tour.sum_travel_durations == 40 + 2 * np.sqrt(200)


def test_tour_reverse(small_tour: Tour):
    small_tour.reverse_section(1, 4)
    assert [r.id_ for r in small_tour.routing_sequence] == ['d0', 'r2', 'r1', 'r0', 'r3', 'd0']


def test_tour_reverse_undo_infeasible_reversal(small_tw_tour: Tour):
    """ensure that upon an unsuccessful attempt of  reversing a section, the original routing sequence and schedules
    are recreated"""
    pre_sequence = [r.id_ for r in small_tw_tour.routing_sequence]
    pre_arrival = small_tw_tour.arrival_schedule[:]
    pre_service = small_tw_tour.service_schedule[:]

    try:
        small_tw_tour.reverse_section(2, 6)
    except InsertionError as e:
        assert pre_sequence == [r.id_ for r in small_tw_tour.routing_sequence] and \
               pre_arrival == small_tw_tour.arrival_schedule and \
               pre_service == small_tw_tour.service_schedule


def test_tour_reverse_InsertionError(small_tw_tour: Tour):
    with pytest.raises(InsertionError):
        small_tw_tour.reverse_section(2, 6)


if __name__ == '__main__':
    pass
