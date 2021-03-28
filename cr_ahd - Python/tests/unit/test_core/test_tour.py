from src.cr_ahd.core_module.tour import Tour
from src.cr_ahd.solving_module.tour_improvement import TwoOpt
import src.cr_ahd.utility_module.utils as ut
import pytest
import numpy as np
import datetime as dt


def test_insert_and_pop(depot_vertex, request_vertices_b):
    distance_matrix = ut.make_travel_dist_matrix([*request_vertices_b, depot_vertex])
    tour = Tour('test', depot_vertex, distance_matrix)
    for request in request_vertices_b:
        tour.insert_and_update(1, request)
    assert all([r.routed for r in request_vertices_b])
    assert abs(tour.sum_travel_duration - ut.travel_time(20 + 3 * np.sqrt(200))) < dt.timedelta(seconds=1)
    for request in request_vertices_b:
        tour.pop_and_update(request.routed[1])
    assert not any([r.routed for r in request_vertices_b])
    assert tour.sum_travel_duration == dt.timedelta(0)


def test_sum_travel_duration(small_criss_cross_tour: Tour):
    assert small_criss_cross_tour.sum_travel_duration == pytest.approx(ut.travel_time(80))


def test_TwoOpt_duration_cost(small_criss_cross_tour: Tour):
    TwoOpt().improve_tour(small_criss_cross_tour)
    assert small_criss_cross_tour.sum_travel_duration == pytest.approx(ut.travel_time(40 + 2 * np.sqrt(200)))


def test_reverse(small_tour: Tour):
    small_tour.reverse_section(1, 4)
    assert [r.id_ for r in small_tour.routing_sequence] == ['d0', 'r2', 'r1', 'r0', 'r3', 'd0']


def test_reverse_undo_infeasible_reversal(small_tw_tour: Tour):
    """ensure that upon an unsuccessful attempt of  reversing a section, the original routing sequence and schedules
    are recreated"""
    pre_sequence = [r.id_ for r in small_tw_tour.routing_sequence]
    pre_arrival = small_tw_tour.arrival_schedule[:]
    pre_service = small_tw_tour.service_schedule[:]

    try:
        small_tw_tour.reverse_section(2, 6)
    except ut.InsertionError as e:
        assert pre_sequence == [r.id_ for r in small_tw_tour.routing_sequence] and \
               pre_arrival == small_tw_tour.arrival_schedule and \
               pre_service == small_tw_tour.service_schedule


def test_tour_reverse_InsertionError(small_tw_tour: Tour):
    with pytest.raises(ut.InsertionError):
        small_tw_tour.reverse_section(2, 6)


if __name__ == '__main__':
    pass
