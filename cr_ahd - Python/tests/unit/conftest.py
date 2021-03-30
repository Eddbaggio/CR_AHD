import datetime as dt

import numpy as np
import pytest

import src.cr_ahd.utility_module.utils as utils
from src.cr_ahd.core_module.carrier import Carrier
from src.cr_ahd.core_module.instance import Instance
from src.cr_ahd.core_module.tour import Tour
from src.cr_ahd.core_module.vehicle import Vehicle
from src.cr_ahd.core_module.vertex import DepotVertex, Vertex


# ==========
# TOURS
# ==========
@pytest.fixture
def small_criss_cross_tour(depot_vertex, request_vertices_a):
    distance_matrix = utils.make_travel_dist_matrix([depot_vertex, *request_vertices_a])
    tour = Tour('small_criss_cross_tour', depot_vertex, distance_matrix)
    for index, request in enumerate(request_vertices_a):
        tour.insert_and_update(index + 1, request)
    return tour


@pytest.fixture
def small_tour(depot_vertex, request_vertices_b):
    distance_matrix = utils.make_travel_dist_matrix([depot_vertex, *request_vertices_b])
    tour = Tour('small_tour', depot_vertex, distance_matrix)
    for index, request in enumerate(request_vertices_b):
        tour.insert_and_update(index + 1, request)
    return tour


@pytest.fixture
def small_tw_tour(depot_vertex, request_vertices_c):
    distance_matrix = utils.make_travel_dist_matrix([depot_vertex, *request_vertices_c])
    tour = Tour('small_tour', depot_vertex, distance_matrix)
    for index, request in enumerate(request_vertices_c):
        tour.insert_and_update(index + 1, request)
    return tour


# ==========
# VERTICES
# ==========

@pytest.fixture
def depot_vertex():
    return DepotVertex('d0', 0, 0)


@pytest.fixture
def request_vertices_a():
    """criss-cross sequence. use to test 2opt"""
    return [
        Vertex('r0', 0, 10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r1', 20, 10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r2', 20, 20, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r3', 10, 20, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r4', 10, 0, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
    ]


@pytest.fixture
def request_vertices_b():
    """used to test reversing part of a tour"""
    return [
        Vertex('r0', 0, 10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r1', 10, 20, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r2', 20, 10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r3', 10, 0, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
    ]


@pytest.fixture
def request_vertices_c():
    """used to test whether tour.reverse_section() handles InsertionError correctly by going back to the original"""
    return [
        Vertex('r0', 0, 10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r1', 10, 20, 0, dt.datetime.min,dt.datetime(year=1, month=1, day=1, hour=0, minute=46)),
        # infeasible if positioned at routing_sequence[5]
        Vertex('r2', 20, 20, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r3', 20, 10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r4', 10, 0, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
    ]


@pytest.fixture
def request_vertices_d():
    """use to test HighestInsertionCostDistance request selection"""
    return [
        Vertex('r0', 10, 10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r1', 20, 10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r2', 20, 20, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r3', 10, 20, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r4', 10, 0, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
    ]


@pytest.fixture
def request_vertices_e():
    """2 clusters of 4"""
    return [
        Vertex('r0', 5, 20, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r4', 20, 5, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r2', 10, 25, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r6', 25, 5, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),

        Vertex('r3', 15, 20, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r1', 10, 15, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r5', 20, 10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r7', 25, 10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
    ]


@pytest.fixture
def request_vertices_f():
    """testing travel times and distances"""
    return [
        DepotVertex('d0', 0, 0),
        Vertex('r0', -10, 0, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r1', -10, -10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r2', 10, -10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r3', 10, 10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r4', 0, 20, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
    ]


@pytest.fixture
def request_vertices_g():
    """all different time windows"""
    return [
        Vertex('r0', -10, 0, 0, dt.datetime(year=1, month=1, day=1, hour=8), dt.datetime(year=1, month=1, day=1, hour=10)),
        Vertex('r1', -10, -10, 0, dt.datetime(year=1, month=1, day=1, hour=10), dt.datetime(year=1, month=1, day=1, hour=12)),
        Vertex('r2', 10, -10, 0, dt.datetime(year=1, month=1, day=1, hour=12), dt.datetime(year=1, month=1, day=1, hour=14)),
        Vertex('r3', 10, 10, 0, dt.datetime(year=1, month=1, day=1, hour=6), dt.datetime(year=1, month=1, day=1, hour=8)),
        Vertex('r4', 0, 20, 0, dt.datetime(year=1, month=1, day=1, hour=16), dt.datetime(year=1, month=1, day=1, hour=18)),
    ]


@pytest.fixture
def request_vertices_random_6():
    """collection of 6 randomly placed vertices [0, 100] with equal tw (0, max) and no service time"""
    return [Vertex(f'r{i}', np.random.randint(0, 100), np.random.randint(0, 100), 0, dt.datetime.min,
                   dt.datetime.max - dt.timedelta(microseconds=1))
            for i in range(6)]


@pytest.fixture
def request_vertices_random_15():
    """collection of 6 randomly placed vertices [0, 100] with equal tw (0, max) and no service time"""
    return [Vertex(f'r{i}', np.random.randint(0, 100), np.random.randint(0, 100), 0, dt.datetime.min,
                   dt.datetime.max - dt.timedelta(microseconds=1))
            for i in range(15)]


# ==========
# VEHICLES
# ==========

@pytest.fixture
def vehicles_5_capacity_0():
    return [Vehicle(f'v{i}', 0) for i in range(5)]


@pytest.fixture
def vehicles_15_capacity_0():
    return [Vehicle(f'v{i}', 0) for i in range(15)]


# ==========
# CARRIERS
# ==========

@pytest.fixture
def carrier_a(depot_vertex, vehicles_5_capacity_0, request_vertices_d):
    dist_matrix = utils.make_travel_dist_matrix([*request_vertices_d, depot_vertex])
    return Carrier('c0', depot_vertex, vehicles_5_capacity_0, request_vertices_d, dist_matrix)


@pytest.fixture
def carrier_b(depot_vertex, vehicles_5_capacity_0, request_vertices_d):
    dist_matrix = utils.make_travel_dist_matrix([*request_vertices_d, depot_vertex])
    return Carrier('c1', depot_vertex, vehicles_5_capacity_0, request_vertices_d, dist_matrix)


@pytest.fixture
def carriers_and_unassigned_requests_3_6(depot_vertex, vehicles_15_capacity_0, request_vertices_random_6):
    dist_matrix = utils.make_travel_dist_matrix([*request_vertices_random_6, depot_vertex])
    carriers = []
    for i in range(3):
        carriers.append(
            Carrier(f'c{i}', depot_vertex, vehicles_15_capacity_0[i * 5:(i + 1) * 5], dist_matrix=dist_matrix))
    return carriers, request_vertices_random_6


@pytest.fixture
def carrier_c(depot_vertex, vehicles_5_capacity_0, request_vertices_g):
    dist_matrix = utils.make_travel_dist_matrix([*request_vertices_g, depot_vertex])
    return Carrier('c1', depot_vertex, vehicles_5_capacity_0, request_vertices_g, dist_matrix)


# ==========
# SUBMITTED REQUESTS
# ==========

@pytest.fixture
def submitted_requests_random_6(request_vertices_random_6):
    length = round(len(request_vertices_random_6) / 2)
    return dict(c1=request_vertices_random_6[:length], c2=request_vertices_random_6[length:])


@pytest.fixture
def submitted_requests_random_15(request_vertices_random_15):
    split = list(utils.split_iterable(request_vertices_random_15, 3))
    return dict(c0=split[0], c1=split[1], c2=split[2])


@pytest.fixture
def submitted_requests_a(request_vertices_e):
    split = list(utils.split_iterable(request_vertices_e, 2))
    return dict(c0=split[0], c1=split[1])


# ==========
# BUNDLE SETS
# ==========

@pytest.fixture
def bundle_set_a(request_vertices_a, depot_vertex, vehicles_5_capacity_0):
    distance_matrix = utils.make_travel_dist_matrix([*request_vertices_a, depot_vertex])
    carrier = Carrier('test', depot_vertex, vehicles_5_capacity_0, dist_matrix=distance_matrix)
    bundle_set = {0: request_vertices_a[:2],
                  1: request_vertices_a[2:3],
                  2: request_vertices_a[3:]}
    return carrier, bundle_set


# ==========
# BIDS
# ==========

@pytest.fixture
def bids_a(carriers_and_unassigned_requests_3_6):
    """random bids on some random bundles. does certainly not represent distance insertion cost!

    :return: dict of bids per bundle per carrier: \n {bundleA: {carrier1: bid, carrier2: bid}, bundleB: {carrier1: bid,
        carrier2: bid} \n Dict[Tuple[Vertex], Dict[Carrier, float]]"""
    carriers, requests = carriers_and_unassigned_requests_3_6
    bids = {
        tuple(requests[:2]): {
            carriers[0]: 9,
            carriers[1]: 7,
            carriers[2]: 5,  # lowest
        },
        tuple(requests[2:4]): {
            carriers[0]: 1,  # lowest
            carriers[1]: 2,
            carriers[2]: 4,
        },
        tuple(requests[4:6]): {
            carriers[0]: 10,
            carriers[1]: 7,  # lowest
            carriers[2]: 11,
        },
    }
    return bids


# ==========
# INSTANCES
# ==========
@pytest.fixture
def instance_a(depot_vertex, carriers_and_unassigned_requests_3_6):
    carriers, requests = carriers_and_unassigned_requests_3_6
    for i, r in enumerate(requests):
        r.carrier_assignment = carriers[int(i / 2)].id_
    return Instance('instance_a', requests, carriers, carriers[0].distance_matrix)
