from src.cr_ahd.core_module.tour import Tour
from src.cr_ahd.core_module.vertex import DepotVertex, Vertex
from src.cr_ahd.solving_module.local_search_visitor import TwoOpt
from src.cr_ahd.utility_module.utils import make_dist_matrix, InsertionError
import pytest
from numpy import sqrt
import matplotlib.pyplot as plt

"""
def create_small_tour():
    # Tour('test_tour', DepotVertex)
    pass


def func(x):
    # tour = create_small_tour()
    return x + 1


def test_answer():
    assert func(3) == 5
"""


#     =============================
@pytest.fixture
def small_criss_cross_tour(depot_vertex, request_vertices_01):
    distance_matrix = make_dist_matrix([depot_vertex, *request_vertices_01])
    tour = Tour('small_criss_cross_tour', depot_vertex, distance_matrix)
    for index, request in enumerate(request_vertices_01):
        tour.insert_and_update(index + 1, request)
    return tour


@pytest.fixture
def small_tour(depot_vertex, request_vertices_02):
    distance_matrix = make_dist_matrix([depot_vertex, *request_vertices_02])
    tour = Tour('small_tour', depot_vertex, distance_matrix)
    for index, request in enumerate(request_vertices_02):
        tour.insert_and_update(index + 1, request)
    return tour


@pytest.fixture
def small_tw_tour(depot_vertex, request_vertices_03):
    distance_matrix = make_dist_matrix([depot_vertex, *request_vertices_03])
    tour = Tour('small_tour', depot_vertex, distance_matrix)
    for index, request in enumerate(request_vertices_03):
        tour.insert_and_update(index + 1, request)
    return tour


@pytest.fixture
def depot_vertex():
    return DepotVertex('d0', 0, 0)


@pytest.fixture
def request_vertices_01():
    """criss-cross sequence. use to test 2opt"""
    return [
        Vertex('r1', 0, 10, 0, 0, 1000),
        Vertex('r2', 20, 10, 0, 0, 1000),
        Vertex('r3', 20, 20, 0, 0, 1000),
        Vertex('r4', 10, 20, 0, 0, 1000),
        Vertex('r5', 10, 0, 0, 0, 1000),
    ]


@pytest.fixture
def request_vertices_02():
    """used to test reversing part of a tour"""
    return [
        Vertex('r1', 0, 10, 0, 0, 1000),
        Vertex('r2', 10, 20, 0, 0, 1000),
        Vertex('r3', 20, 10, 0, 0, 1000),
        Vertex('r4', 10, 0, 0, 0, 1000),
    ]


@pytest.fixture
def request_vertices_03():
    """used to test whether tour.reverse_section() handles InsertionError correctly by going back to the original"""
    return [
        Vertex('r1', 0, 10, 0, 0, 200),
        Vertex('r2', 10, 20, 0, 20, 55),  # infeasible if positioned at routing_sequence[5]
        Vertex('r3', 20, 20, 0, 0, 200),
        Vertex('r4', 20, 10, 0, 0, 200),
        Vertex('r5', 10, 0, 0, 0, 200),
    ]


def test_tour_cost(small_criss_cross_tour: Tour):
    assert small_criss_cross_tour.cost == 80


def test_TwoOpt_cost(small_criss_cross_tour: Tour):
    TwoOpt().finalize_tour(small_criss_cross_tour)
    assert small_criss_cross_tour.cost == 40 + 2 * sqrt(200)


def test_tour_reverse(small_tour: Tour):
    small_tour.reverse_section(1, 4)
    assert [r.id_ for r in small_tour.routing_sequence] == ['d0', 'r3', 'r2', 'r1', 'r4', 'd0']


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
